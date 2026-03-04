    async def serialize_to_json(self, data: Any) -> str:
        """Converteert complexe Python-objecten (zoals dataclasses, UUID, datetime)
        naar een JSON-string. Geschikt voor async-contexten."""

        async def to_serializable(obj: Any) -> Any:
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            # Gebruik isoformat als het beschikbaar is (voor datetime e.d.)
            if hasattr(obj, "isoformat") and callable(obj.isoformat):
                return obj.isoformat()
            # Gebruik hex als het beschikbaar is (voor UUID e.d.)
            if hasattr(obj, "hex"):
                return str(obj)
            if is_dataclass(obj):
                return {k: await to_serializable(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                # Paralleliseer lijstverwerking
                return await asyncio.gather(*(to_serializable(i) for i in obj))
            if isinstance(obj, dict):
                keys = list(obj.keys())
                values = await asyncio.gather(*(to_serializable(obj[k]) for k in keys))
                return dict(zip(keys, values))
            if hasattr(obj, "__dict__"):
                return {k: await to_serializable(v) for k, v in obj.__dict__.items()}
            return str(obj)

        # Simuleer async gedrag (handig als je later b.v. async IO toevoegt)
        return json.dumps(await to_serializable(data), indent=2, default=str)

    async def filter_high(
        self,
        data_json: str,
    ) -> None:
        """
        Filtert entries uit een JSON-string op similarity
        en geeft een JSON-string terug met alleen de 'content' velden.
        """
        try:
            # Zet de JSON-string om naar Python object
            data = json.loads(data_json)
        except json.JSONDecodeError as e:
            _LOGGER.error(f"Fout bij JSON parsing: {e}")
            return json.dumps({"contents": []}, indent=2)

        # Zorg dat data altijd een lijst is
        if not isinstance(data, list):
            data = [data]

        filtered_contents = []

        for item in data:
            # Probeer eerst gist_data, dan event_data
            content = None
            if "gist_data" in item and isinstance(item["gist_data"], dict):
                content = item["gist_data"].get("content")
            elif "event_data" in item and isinstance(item["event_data"], dict):
                content = item["event_data"]

            if content:
                filtered_contents.append(content)

        # Maak er JSON van
        return json.dumps({"contents": filtered_contents}, indent=2)

    async def async_memobase_insert(self, user_input, last_assistant_output) -> None:
        """Insert Blob into MemoaBase Database"""
        u = await self.mb_client.get_or_create_user(string_to_uuid("Ollama"))
        try:
            blob = ChatBlob(
                messages=[
                    {"role": "user", "content": user_input.text},
                    {"role": "assistant", "content": last_assistant_output},
                ]
            )

            # Asynchronous insert (default behavior)
            await u.insert(blob)

            # Asynchronous flush (default behavior)
            await u.flush()
        except Exception as err:
            _LOGGER.warning(f"MemoBase add_texts failed: {err}")
