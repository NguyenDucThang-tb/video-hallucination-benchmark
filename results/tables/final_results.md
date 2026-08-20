| Models | Training-free | VidHalluc |  |  |  |  | VideoHallucer |  |  |  |  |  | EventHallusion |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  |  | BQA | MCQ | STH | TSH | AVG | ORH | TPH | SDH | EFH | ENFH | AVG | AVG |
| LLaVA-OV-7B - Base | Yes | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| LLaVA-OV-7B - TCD | Yes | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| LLaVA-OV-7B - DINO-HEAL | Yes | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| LLaVA-OV-7B - SEASON | Yes | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Qwen2.5-VL-7B - Base | Yes | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Qwen2.5-VL-7B - TCD | Yes | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Qwen2.5-VL-7B - DINO-HEAL | Yes | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Qwen2.5-VL-7B - SEASON | Yes | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| LLaVA-Video-7B - Base | Yes | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| LLaVA-Video-7B - TCD | Yes | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| LLaVA-Video-7B - DINO-HEAL | Yes | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| LLaVA-Video-7B - SEASON | Yes | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

All values are percentages. N/A means not executed, incomplete, or unsupported.
VidHalluc AVG is the strict macro-average of BQA/MCQ/official STH/TSH; VideoHallucer AVG is the macro-average of five strict pair accuracies; EventHallusion AVG is sample-weighted binary accuracy over all configured splits.
