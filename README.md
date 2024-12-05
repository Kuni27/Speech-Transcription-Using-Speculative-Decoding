# Enhancing Speech Transcription Using Speculative Decoding for Real-Time Performance

## Objective
The goal of this project is to enhance real-time multilingual speech transcription by integrating speculative decoding into the Whisper-v3 model. This addresses latency issues in traditional speech recognition systems, ensuring faster transcription without compromising accuracy, particularly in dynamic live environments like conversations, captions, or voice assistants.

---

## Key Features
- **Speculative Decoding**: Implemented a student-teacher architecture to reduce latency while maintaining high transcription accuracy.
- **Real-Time Performance**: Achieved ~3x faster transcription compared to traditional decoding in Whisper-large-v3.
- **Multilingual Support**: Trained a custom student model compatible with Whisper-large-v3 to support multiple languages.

---

## Methodology
1. **Architecture Design**:
   - Faster assistant (student) model predicts tokens, verified by the main (teacher) model.
   - Shared encoder structure ensures compatibility.
2. **Dataset Preprocessing**:
   - Utilized [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1) and [VoxPopuli](https://huggingface.co/datasets) datasets for multilingual training.
   - Generated pseudo-labels for training using `run_pseudo_labelling.py`.
3. **Training**:
   - Encoders frozen from the teacher model.
   - Reduced decoder layers from 32 to 2.
   - Performed optimization steps on 150k samples across 8 languages using TAMU HPRC.

---

## Results

### Word Error Rate (WER) Comparison
| Dataset              | Teacher Model WER | Student Model WER |
|----------------------|-------------------|-------------------|
| Italian Common Voice | 6.10%            | 5.93%            |
| Dutch Common Voice   | 4.27%            | 4.32%            |
| Hungarian Common Voice | 12.94%         | 12.28%           |
| Romanian Common Voice | 10.00%          | 4.19%            |
| Polish Common Voice   | 5.75%           | 5.52%            |
| Slovenian Common Voice | 11.51%         | 11.23%           |
| Slovak Common Voice   | 17.91%          | 15.83%           |
| Finnish Voxpopuli     | 13.69%          | 13.09%           |
| Italian Voxpopuli     | 24.92%          | 22.93%           |
| Dutch Voxpopuli       | 15.24%          | 15.23%           |
| Hungarian Voxpopuli   | 17.49%          | 16.56%           |
| Romanian Voxpopuli    | 15.37%          | 14.85%           |
| Polish Voxpopuli      | 10.10%          | 9.74%            |
| Slovenian Voxpopuli   | 36.46%          | 37.00%           |
| Slovak Voxpopuli      | 13.96%          | 11.66%           |
| French Voxpopuli      | 11.97%          | 11.47%           |

From the results above, the speculative decoding approach demonstrates similar or improved Word Error Rates (WER) while significantly reducing latency.

---

## Challenges
- Incompatibility of existing models with Whisper-large-v3.
- High computational and memory requirements.
- Debugging in the High-Performance Computing (HPRC) environment.

---

## Getting Started

### Installation
Clone the repository:
```bash
git clone https://github.com/Kuni27/Speech-Transcription-Using-Speculative-Decoding.git
```

### Running the Demo
Start the transcription servers by executing the following commands:

```bash
streamlit run whisper_v3_transcription.py --server.port 8080
streamlit run speculative_decoding.py --server.port 8501
```
### Requirements

Before you begin, ensure you have the following installed:
pip install the required packages from the [setup.py](./setup.py) file:
```bash
pip install -e .
```
