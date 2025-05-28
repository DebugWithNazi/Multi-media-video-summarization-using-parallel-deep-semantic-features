[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/supervised-video-summarization-via-multiple/supervised-video-summarization-on-summe)](https://paperswithcode.com/sota/supervised-video-summarization-on-summe?p=supervised-video-summarization-via-multiple)

# MSVA (Multi Source Visual Attention) — Enhanced Version

This is the enhanced and updated version of the **MSVA** (Multi Source Visual Attention) model for **Supervised Video Summarization**, incorporating significant improvements proposed in:

> **Nazia Ramzan**  
> *Multi-Media Video Summarization Using Parallel Deep Semantic Features*  
> University of Engineering and Technology, Lahore, Pakistan

The original MSVA paper is available at:  
- [arXiv](https://arxiv.org/pdf/2104.11530.pdf)

---

## 📌 Key Enhancements by Nazia Ramzan

- Parallel feature extraction via dual pipelines (motion & spatial).
- Use of **Inception-V1** and **ResNet-101** for spatial feature augmentation.
- Modified MSVA architecture with improved F1-score on benchmark datasets.
- End-to-end pipeline including **H5 summary to video conversion**.
- Added dropout tuning and extra layers for feature richness.
- Improved evaluation using subjective criteria: **fluency**, **pace**, **content quality**.

> 📈 **F1 Score Achievements:**  
> - **SumMe**: 64 (vs. 54.5 original)  
> - **TvSum**: 64 (vs. 62.5 original)

---

## 🧠 Abstract

This project presents a **deep learning framework** for summarizing long videos by extracting meaningful content using **parallel deep semantic features**. The summarization process uses multiple modalities (RGB, Flow, and Object-based features) and feeds them into a unified attention-based MSVA model, yielding human-like condensed video summaries.

---

## 🗂 Dataset and Features

### 📥 Download

```bash
wget -O datasets.tar https://zenodo.org/record/4682137/files/msva_video_summarization.tar
tar -xvf datasets.tar
```

### 📦 Feature Extraction

- **Motion Features**: I3D pretrained model (RGB & Flow).
- **Spatial Features**: ResNet-101 and Inception-V1 via GoogleNet pool5.
- Combined and stored in `.h5` format.

---

## 🚀 Setup

```bash
git clone git@github.com:VideoAnalysis/MSVA.git
cd MSVA
conda env create -f environment.yml
conda activate Multi-media-video-summarization
```

---

## 🏋️‍♂️ Training

```bash
python train.py -params parameters.json
```

---

## 🔍 Inference

```bash
python inference.py -dataset "summe" -video_name "Air_Force_One" -model_weight "model_weights/summe_updated_model.tar.pth"
```

---

## 🎬 Summary Video Generation

```bash
python video_from_summary.py --frames_path ./video_frames/ --summary_file ./summary.h5
```

---

## 🔧 Sample Configuration (`parameters.json`)

```json
{
  "max_summary_length": 0.15,
  "weight_decay": 0.00001,
  "epochs_max": 300,
  "train_batch_size": 5,
  "fusion_technique": "inter",
  "method": "mean",
  "sample_technique": "sub",
  "stack": "v",
  "apertures": [250],
  "combis": [[1,1,1]],
  "feat_input": {
    "feature_size": 365,
    "L1_out": 365,
    "L2_out": 365,
    "L3_out": 512,
    "pred_out": 1,
    "apperture": 250,
    "dropout1": 0.5,
    "att_dropout1": 0.5,
    "feature_size_1_3": 1024,
    "feature_size_4": 365
  }
}
```

---

## 📊 Evaluation Metrics

Evaluated with:
- User feedback (fluency, pace, content management)
- Subjective and quantitative precision/recall
- Benchmark comparison with MSVA, MAVS, VASNet, etc.

---

## 🏆 Citation

If you use this work, please cite both original and extended contributions:

```bibtex
@article{ghauri2021MSVA, 
   title={Supervised Video Summarization via Multiple Feature Sets with Parallel Attention},
   author={Ghauri, Junaid Ahmed and Hakimov, Sherzod and Ewerth, Ralph}, 
   journal={IEEE ICME}, 
   year={2021}
}

```

---

## 🙏 Acknowledgements

We gratefully acknowledge the enhancements introduced by **Nazia Ramzan** and her co-authors, whose work advanced MSVA into a complete, high-performing, and modular framework for modern video summarization tasks.

---

## 🔗 Related Resources

- [SumMe Dataset](https://gyglim.github.io/me/vsum/index.html)
- [TVSum Dataset](https://github.com/yalesong/tvsum)
- [Kinetics Dataset / I3D Info](https://github.com/deepmind/kinetics-i3d)
- [OpenCV Video Processing](https://docs.opencv.org/)
