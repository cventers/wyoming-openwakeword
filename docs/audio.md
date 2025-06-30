# Audio Pipeline

This document describes how audio data flows through the `wyoming-openwakeword` server and how wake word models are used during detection.

## Overview

`wyoming-openwakeword` is a [Wyoming protocol](https://github.com/rhasspy/wyoming) service. It does **not** capture audio on its own. Instead, any device that can speak the Wyoming protocol (for example, another program running on the same machine or a completely different host) connects to the server and streams audio events. Clients send `AudioStart`, followed by one or more `AudioChunk` events and finally `AudioStop`. The server replies with `Detection` events when a wake word is found, or with `NotDetected` when the audio stream finishes without a match.

Audio may come from microphones, prerecorded files or even network streams. As long as the client formats the data using the Wyoming audio messages, the server can run on one device while the microphone lives on another.

## Receiving Audio

When a client connects, the handler stores per‑client buffers. Incoming `AudioChunk` messages are converted to 16‑bit mono at 16kHz using `AudioChunkConverter` and appended to a ring buffer. The relevant code lives in `OpenWakeWordEventHandler.handle_event`:

```python
self.data.audio[: -len(chunk_array)] = self.data.audio[len(chunk_array) :]
self.data.audio[-len(chunk_array) :] = chunk_array
self.data.new_audio_samples = min(
    len(self.data.audio),
    self.data.new_audio_samples + len(chunk_array),
)
```

【F:wyoming_openwakeword/handler.py†L116-L126】

After buffering, the handler releases a semaphore so that the mel‑spectrogram thread knows that new audio is ready.

## Audio Processing Threads

The server uses three processing stages that run in separate threads:

1. **Audio → Mel Spectrograms** – `mels_proc`
2. **Mel Spectrograms → Embedding Features** – `embeddings_proc`
3. **Embeddings → Wake Word Probabilities** – `ww_proc` (one thread per loaded model)

### Mel Spectrograms

The first stage loads the bundled `melspectrogram.tflite` model and waits for audio. Once enough samples accumulate it creates a batch and runs the model. The resulting mel windows are stored for each client:

```python
melspec_model.invoke()
mels = melspec_model.get_tensor(melspec_output_index)
...  # add windows to client buffers
```

【F:wyoming_openwakeword/openwakeword.py†L94-L100】

### Embedding Features

The second stage converts mel windows into embedding vectors using `embedding_model.tflite`. Batches are created in the same way and the embeddings are appended to per‑wake‑word buffers:

```python
embedding_model.set_tensor(embedding_input_index, mels_tensor)
embedding_model.invoke()
embeddings = embedding_model.get_tensor(embedding_output_index)
```

【F:wyoming_openwakeword/openwakeword.py†L190-L196】

### Wake Word Models

For each loaded wake word a dedicated thread runs `ww_proc`. These threads read embedding windows and run the associated TFLite model. If the probability exceeds the configured threshold often enough (based on `trigger_level`), a `Detection` event is sent to the client:

```python
if probability.item() >= client_data.threshold:
    client_data.activations += 1
    if client_data.activations >= client_data.trigger_level:
        client_data.is_detected = True
        client_data.activations = 0
        coros.append(
            client.event_handler.write_event(
                Detection(
                    name=ww_model_key,
                    timestamp=todo_timestamps[client_id],
                ).event()
            ),
        )
```

【F:wyoming_openwakeword/openwakeword.py†L348-L361】

## Switching Devices

Because the service only communicates via the Wyoming protocol, audio can originate on one host and detection can happen on another. The server simply listens on the URI specified with `--uri`, such as `tcp://0.0.0.0:10400` for network access or `stdio://` for integration with another process.

A separate microphone program can stream audio to the server, and multiple clients can connect concurrently. Each client maintains its own buffers and wake word state so detections do not interfere with one another.

## Model Usage

Wake word models are stored as `.tflite` files. Built‑in models live in `wyoming_openwakeword/models`. Additional directories can be supplied with `--custom-model-dir`. Models may also be preloaded at startup using `--preload-model`. When a `Detect` command is received from a client the handler ensures that the requested models are loaded; if necessary it starts a new `ww_proc` thread for each model. The function `ensure_loaded` handles this:

```python
state.wake_words[model_key] = WakeWordState()
state.ww_threads[model_key] = Thread(
    target=ww_proc,
    daemon=True,
    args=(state, model_key, model_path, asyncio.get_running_loop()),
)
state.ww_threads[model_key].start()
```

【F:wyoming_openwakeword/handler.py†L246-L259】

Model names normally include a version suffix (for example `hey_jarvis_v0.1`). The handler exposes human‑readable names and phrases through the Wyoming `Describe` message so clients know which models are available.

## When No Wake Word is Found

Once a client sends `AudioStop`, the handler waits until all pending audio has been processed. If none of the active wake words were detected a `NotDetected` message is returned:

```python
if not any(
    ww_data.is_detected for ww_data in self.data.wake_words.values()
):
    await self.write_event(NotDetected().event())
```

【F:wyoming_openwakeword/handler.py†L151-L155】

## Summary

Audio flows into `wyoming-openwakeword` entirely through Wyoming protocol messages. Dedicated threads transform the stream into mel spectrograms, embeddings and finally wake word probabilities via TensorFlow Lite models. Because the service only processes events, audio can originate from any device capable of speaking the protocol, and models may be added or removed dynamically.
