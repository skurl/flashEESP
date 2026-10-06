# EmbeddedFlashEE

[flashEE](https://github.com/skurl/flashee) running on an ESP32-S3-DevKitC-1 N16R8. Send a protein sequence over USB or WiFi, get a 320-d embedding back.

# Video Overview

[![EmbeddedFlashEE video](https://img.youtube.com/vi/sB1c4Pdw5B0/0.jpg)](https://youtu.be/sB1c4Pdw5B0)

# Deployment

Plug the board in (the COM port), with [ESP-IDF v5.3](https://docs.espressif.com/projects/esp-idf/en/v5.3/esp32s3/get-started/) installed.

Add your WiFi, then flash firmware + model:

```bash
cp esp-tflm/main/wifi_secrets.h.example esp-tflm/main/wifi_secrets.h   # fill in SSID/password
source ~/esp/esp-idf/export.sh && cd esp-tflm && idf.py -p /dev/cu.usbmodem* flash
```

Create Embeddings:

```bash
python3 embed.py MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG 192.168.0.155
```

```python
from embed import embed
v = embed("MQIFVKTL...", "192.168.0.155")   # list of 320 floats (mean-pooled)
```

Up to 510 residues per sequence.

# How efficient is it?

Not much.

| ubiquitin (76 aa) | time |
|---|---|
| ESP32-S3 | 61 s |
| MacBook M4, PyTorch | 3.9 ms (~15,000× faster) |

# Is it accurate?

Pretty much lossless. Using cosine similarity, the boards embeddings using int8 weigths matches fp32 at 0.980–0.998 across different length proteins.


# Acknowledgements

Thank you Anthropic for providing free access to Claude models as part of the iGEM competition.

Maciej Robert Szczesny 2026
