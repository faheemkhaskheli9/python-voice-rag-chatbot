# Architecture Notes: Python Voice RAG Chatbot

## Pipeline

```text
Microphone -> Speech Recognition -> RAG -> LLM -> TTS -> Speaker Output
```

## Components

- Microphone input capture
- Speech recognition
- RAG-based retrieval
- LLM response generation
- Text-to-speech output

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
