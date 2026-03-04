"""The conversation platform for the Ollama integration with MemoBase and Acontext."""

from __future__ import annotations

from typing import Literal
import asyncio
import logging

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import OllamaConfigEntry
from .const import CONF_PROMPT, DOMAIN
from .entity import OllamaBaseLLMEntity

from memobase import AsyncMemoBaseClient
from memobase.core.blob import ChatBlob
from memobase.utils import string_to_uuid

from .acontext import AcontextAsyncClient

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OllamaConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [OllamaConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class OllamaConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    OllamaBaseLLMEntity,
):
    """Ollama conversation agent with MemoBase and Acontext memory."""

    _attr_supports_streaming = True

    def __init__(self, entry: OllamaConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )
            
        # Initialize clients once
        self.acontext = AcontextAsyncClient(
            api_key="sk-ac-your-root-api-bearer-token",
            base_url="http://127.0.0.1:8029/api/v1",
        )

        self.mb_client = AsyncMemoBaseClient(
            project_url="http://127.0.0.1:8020",
            api_key="secret",
        )
        
        # Map HA conversation IDs to Acontext session IDs
        self._session_map: dict[str, str] = {}

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL
        
    async def _get_or_create_acontext_session(self, conversation_id: str | None) -> str:
        """Get existing Acontext session or create a new one."""
        if conversation_id and conversation_id in self._session_map:
            return self._session_map[conversation_id]

        session = await self.acontext.sessions.create()
        # Optionally create space/disk here if required per session
        # disk = await self.acontext.disks.create()
        
        if conversation_id:
            self._session_map[conversation_id] = session.id
        return session.id

    async def async_memobase_insert(self, user_text: str, assistant_text: str) -> None:
        """Insert Blob into MemoBase Database."""
        try:
            u = await self.mb_client.get_or_create_user(string_to_uuid("Ollama"))
            blob = ChatBlob(
                messages=[
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            )
            await u.insert(blob)
            await u.flush()
        except Exception as err:
            _LOGGER.warning(f"MemoBase add_texts failed: {err}")

    async def async_acontext_update(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Store messages in Acontext and trigger a background flush."""
        try:
            # Store user message
            await self.acontext.sessions.store_message(
                session_id=session_id,
                blob={"role": "user", "content": user_text},
            )
            
            # Store assistant message
            await self.acontext.sessions.store_message(
                session_id=session_id,
                blob={"role": "assistant", "content": assistant_text},
            )
            
            # Background flush so HA isn't blocked waiting for processing
            asyncio.create_task(self.acontext.sessions.flush(session_id))
            
        except Exception as err:
            _LOGGER.warning(f"Acontext update failed: {err}")

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API with MemoBase and Acontext memory."""
        settings = {**self.entry.data, **self.subentry.data}

        # 1. Map HA Conversation to Acontext Session
        session_id = await self._get_or_create_acontext_session(user_input.conversation_id)

        # 2. Fetch MemoBase Context
        memobase_context_text = ""
        try:
            u = await self.mb_client.get_or_create_user(string_to_uuid("Ollama"))
            mb_context = await u.context(
                max_token_size=5000,
                event_similarity_threshold=0.4,
                require_event_summary=True,
                chats=[{"role": "user", "content": user_input.text}],
                time_range_in_days=90,
            )
            memobase_context_text = mb_context
            _LOGGER.warning(f"MemoBase Check: {memobase_context_text}")

        except Exception as err:
            _LOGGER.warning(f"Failed to fetch MemoBase context: {err}")

        # 3. Fetch Acontext Summary
        acontext_summary_text = ""
        try:
            summary = await self.acontext.sessions.get_session_summary(session_id=session_id)
            if summary:
                acontext_summary_text = str(summary)
        except Exception as err:
            _LOGGER.warning(f"Failed to fetch Acontext summary: {err}")

        # 4. Build the final System Prompt
        system_prompt = user_input.extra_system_prompt or ""
        
        if memobase_context_text:
            system_prompt += f"\n\n### MemoBase Memory Context\n{memobase_context_text}"
            
        if acontext_summary_text:
            system_prompt += f"\n\n### Acontext Session State\n{acontext_summary_text}"

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                settings.get(CONF_LLM_HASS_API),
                settings.get(CONF_PROMPT),
                user_input.extra_system_prompt,
                system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # 5. Generate the response
        await self._async_handle_chat_log(chat_log)
        result = conversation.async_get_result_from_chat_log(user_input, chat_log)
        
        # Link the generated/existing HA conversation_id so it continues properly
        if not result.conversation_id:
            result.conversation_id = session_id
            self._session_map[session_id] = session_id

        # 6. Extract Assistant Output and Save to Memory
        assistant_messages = [
            item.content
            for item in chat_log.content
            if getattr(item, "role", None) == "assistant"
        ]

        if assistant_messages:
            last_assistant_output = assistant_messages[-1]
            
            # Update Acontext
            await self.async_acontext_update(session_id, user_input.text, last_assistant_output)
            
            # Update MemoBase
            asyncio.create_task(self.async_memobase_insert(user_input.text, last_assistant_output))
        else:
            _LOGGER.warning("Geen assistant output gevonden, geheugen is niet bijgewerkt.")

        return result
