# Carnegie Mellon University: AI & Vision Courses Reference

A reference of CMU's AI, machine learning, and computer vision courses relevant to the **AI Hardware Engineer Roadmap** — compiled from official course pages and catalogs.

**Sources:**
- [07-280 AI & ML I](https://www.cs.cmu.edu/~07280/#schedule)
- [15-463 Computational Photography](https://graphics.cs.cmu.edu/courses/15-463/2018_fall/)
- [16-385 Computer Vision Spring 2026 — Lectures](https://16385.courses.cs.cmu.edu/spring2026/lectures)
- [Szeliski: Computer Vision (2nd ed.)](https://szeliski.org/Book/)

---


## 1. 07-280: AI & Machine Learning I

**New in Spring 2026** — Replaces 15-281 and 10-315. Foundation for 07-380 AI & ML II.

### Overview

| Item | Details |
|------|---------|
| **Lectures** | Tue + Thu, 11:00 am–12:20 pm, Tepper 1403 |
| **Recitation** | Friday afternoon (5 sections) |
| **Instructors** | Nihar Shah, Pat Virtue |
| **Education Associate** | Brynn Edmunds |
| **Textbook** | No required textbook; readings from AIMA, Bishop, Daume, Goodfellow, MML, Mitchell, Murphy, KMPA (all online or via CMU Library) |

### Course Description

Integrated introduction to AI and ML bridging core methods with modern approaches. Students build implementations of landmark systems: **AlexNet**, **GPT-2**, and **AlphaZero**. Covers ethics and responsible AI development.

### Grading

| Component | Weight |
|-----------|--------|
| Midterm 1 | 15% |
| Midterm 2 | 15% |
| Final Exam | 25% |
| Programming/Written Homework | 30% |
| Online Homework | 5% |
| Pre-reading Checkpoints | 5% |
| Participation (in-class polls) | 5% |

**Grade boundaries (rough):** A >=90%, B 80-90%, C 70-80%, D 60-70%. Not curved.

### Prerequisites (Strict)

- **15-122** Principles of Imperative Computation
- **Probability** (concurrent)
- **Linear Algebra** (prior)
- **15-151** or **21-127** Mathematical Foundations (prior)
- **Calculus 2** (concurrent)

### Schedule (Spring 2026)

| Dates | Topic |
|-------|-------|
| 1/13 | 1. Introduction |
| 1/15 | 2. Heuristic Search |
| 1/20 | 3. Adversarial Search |
| 1/22 | 4. Constraint Satisfaction Problems |
| 1/27 | 5. ML Problem Formulation |
| 1/29 | 6. Decision Trees |
| 2/3 | 7. Linear Regression |
| 2/5 | 8. Optimization |
| 2/10 | 9. Logistic Regression |
| 2/12 | 10. Feature Engineering and Regularization |
| 2/17 | 11. Neural Networks |
| 2/19 | 12. Neural Networks (cont.) |
| **2/24** | **Midterm Exam 1** |
| 2/26 | 13. AI Alignment |
| 3/3, 3/5 | Spring Break |
| 3/10 | 14. PyTorch, Autograd, Pre-training/Transfer/Fine-tuning |
| 3/12 | 15. Deep Learning for Computer Vision, GPUs |
| 3/17 | 16. MLE and Probabilistic Modeling |
| 3/19 | 17. NLP, Markov Chains, N-grams |
| 3/24 | 18. Feature Learning, Word Embeddings |
| 3/26 | 19. NLP: Attention, Position Encoding |
| 3/31 | 20. Transformers, LLMs |
| 4/2 | 21. Markov Decision Processes |
| 4/7 | 22. Reinforcement Learning |
| 4/9 | Carnival (no class) |
| 4/14 | 23. Deep Reinforcement Learning |
| 4/16 | 24. Monte Carlo Tree Search |
| **4/21** | **Midterm Exam 2** |
| 4/23 | 25. AI/ML Ethics |

### Assignments

| HW | Type | Due |
|----|------|-----|
| HW0 | Online | 1/15 Thu |
| HW1 | Online, Written, Programming | 1/22 Thu |
| HW2 | Online, Written, Programming (Search & Games) | 1/29 Thu |
| HW3 | Online only | 2/5 Thu |
| HW4 | Written only | 2/12 Thu |
| HW5 | Mostly Programming | 2/19 Thu |
| HW6 | Mostly Written | 2/26 Thu |
| HW7 | Mostly Programming | 3/12 Thu |
| **HW8** | **Building AlexNet** | 3/19 Thu |
| HW9 | Online, Written, Programming | 3/26 Thu |
| HW10 | Online, Written, Programming | 4/2 Thu |
| **HW11** | **Building GPT-2** | 4/16 Thu |
| **HW12** | **Building AlphaZero** | 4/23 Thu |

### Policies

- **Late days:** 6 total across all assignments; max 2 per assignment
- **Pre-reading:** Lowest 2 checkpoints dropped
- **Participation:** >=80% of in-class polls for full credit
- **Collaboration:** Conceptual discussion allowed; no sharing code/text; generative AI may not be used to generate submissions
- **Programming partners:** Groups of 2 allowed for programming components only

### Comparison: 07-280 vs 10-301

| 07-280 | 10-301 |
|--------|--------|
| Heuristic Search, Adversarial Search, CSPs | — |
| ML Parallelism/GPU Basics | — |
| Monte Carlo Tree Search | — |
| Transformer networks, LLMs | Y |
| Reinforcement Learning | Y |
| ML fundamentals (decision trees -> neural nets) | Y |
| Fulfills AI Major core | Y |
| Prereq for 07-380 AI & ML II | Y |

---

## 2. 15-463: Computational Photography — Deep Insight Before Computer Vision

**Recommended before 16-385 Computer Vision.** Provides foundational understanding of imaging physics, camera pipelines, and computational methods that underpin both graphics and vision.

*Source: [15-463 Fall 2018](https://graphics.cs.cmu.edu/courses/15-463/2018_fall/)*

### Overview

| Item | Details |
|------|---------|
| **Cross-listing** | 15-463 (undergrad), 15-663 (Master's), 15-862 (PhD) |
| **Schedule** | Mon + Wed, 12:00-1:20 PM |
| **Textbook** | [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/) (Szeliski), free online |

### Course Description

Computational photography is the convergence of **computer graphics**, **computer vision**, and **imaging**. It overcomes traditional camera limitations by combining imaging and computation for new ways of capturing, representing, and interacting with the physical world.

Topics: modern image processing pipelines (mobile/DSLR), image/video editing, 3D scanning, coded photography, lightfield imaging, time-of-flight, VR/AR displays, computational light transport. Advanced topics: cameras at light speed, non-line-of-sight imaging, seeing through tissue.

### Prerequisites (one of)

- 18-793 Image and Video Processing
- **15-462 Computer Graphics**, OR
- 16-720 Computer Vision, OR
- **16-385 Computer Vision**

Linear algebra, calculus, programming, and image computation required.

### Grading

| Component | Weight |
|-----------|--------|
| 7 Homework Assignments | 70% |
| Final Project | 25% |
| Class Participation | 5% |

**Late policy:** 6 free late days total; each additional late day = 10% penalty; max 4 days late per assignment.

### Syllabus (Fall 2018)

| Topic |
|-------|
| Introduction |
| Digital photography pipeline |
| Pinholes and lenses |
| Photographic optics and exposure |
| High dynamic range imaging |
| Tonemapping and bilateral filtering |
| Color |
| Image compositing |
| Gradient-domain image processing |
| Focal stacks and lightfields |
| Deconvolution |
| Camera models and calibration |
| Two-view geometry |
| Radiometry and reflectance |
| Photometric stereo |
| Light transport matrices |
| Computational light transport |
| Stereo and structured light |
| Time-of-flight imaging |
| Non-line-of-sight imaging |
| Fourier optics |
| Monte Carlo rendering 101 |

### Assignments

7 homework assignments with **programming (Matlab)** and **photography (DSLR)** components. Final project may use lightfield cameras, ToF cameras, depth sensors, structured light systems.

### Why Before Computer Vision

15-463 builds intuition for **how images are formed** (optics, radiometry, sensors) and **how to process them** (HDR, deconvolution, calibration, stereo). This physical and algorithmic foundation makes 16-385 Computer Vision (detection, recognition, geometry) much easier to grasp.

---

## 3. Graphics & Imaging Courses

| Code | Name | Notes |
|------|------|-------|
| **15-462** | Computer Graphics | Rendering, transforms, shading — foundation for 15-463 |
| **15-463** | Computational Photography | Imaging physics, camera pipelines, HDR, lightfields — **deep insight before 16-385** |
| **16-385** | Computer Vision | Image processing, detection, recognition, geometry-based vision |

---

## 4. Self-Study Mapping to Roadmap

For learners following the **AI Hardware Engineer Roadmap**, here is how CMU's AI/vision curriculum aligns:

| Roadmap Phase | CMU Course(s) | Overlap |
|---------------|---------------|---------|
| **Phase 3: Neural Networks** | 07-280 (Neural Nets, AlexNet, PyTorch) | Graphs, training, autodiff before Phase 4 Track A/B hardware |
| **Phase 3: Computer Vision** | **15-463** (Computational Photography) -> 16-385 (Computer Vision) | Imaging physics, camera pipelines, then detection/recognition |
| **Phase 3: Edge AI** | 07-280 (deployment themes), Jetson-adjacent labs | On-device pipeline, latency/privacy context; pairs with Phase 4 Track B |
| **Phase 3: Sensor Fusion** | 15-463 (imaging), 16-385 (Vision) | Multi-sensor perception before Phase 4 Jetson integration |
| **Phase 4 Track B (Jetson)** | 07-280, 16-385, Jetson/Holoscan labs | Models on device, pipelines, latency |
| **Phase 5: Edge Computing** | 07-280, 16-385, Jetson/Holoscan | Efficient models, streaming, Holoscan |
| **Phase 5: AI Chip Design** | 07-280 (optimization, GPU/parallel), 16-211 (math) | Parallel compute, linear algebra for accelerators |

### Suggested Self-Study Order (CMU-Inspired, AI focus)

1. **15-122 equivalent** — Imperative programming (C/Python)
2. **21-120, 21-122, 21-241** — Calculus, linear algebra
3. **07-280 topics** — Search -> ML -> Neural Nets -> RL -> Transformers
4. **15-462** (optional) — Computer Graphics — rendering, transforms
5. **15-463** — **Computational Photography** — imaging physics, camera pipelines, HDR, lightfields *(deep insight before vision)*
6. **16-385** — Computer vision

---

## 5. Additional Resources: Szeliski Book & Related Courses

### Computer Vision: Algorithms and Applications (2nd ed.)

**[https://szeliski.org/Book/](https://szeliski.org/Book/)** — Richard Szeliski, University of Washington (c 2022)

The canonical computer vision textbook. Free PDF download for personal use. Used by 15-463 Computational Photography and many vision courses worldwide. Covers image formation, feature detection, stereo, structure from motion, recognition, and more.

### Related Courses (from [Szeliski Book](https://szeliski.org/Book/))

Additional good sources for computer vision and computational photography, sorted roughly by most recent:

| Course | Institution | Instructor(s) | Term |
|--------|-------------|---------------|------|
| [CS5670 Introduction to Computer Vision](https://www.cs.cornell.edu/courses/cs5670/2025sp/) | Cornell Tech | Noah Snavely | Spring 2025 |
| [6.8300/6.8301 Advances in Computer Vision](https://szeliski.org/Book/) | MIT | Bill Freeman, Antonio Torralba, Phillip Isola | Spring 2023 |
| [16-385 Computer Vision](http://www.cs.cmu.edu/~16385/) | CMU | Matthew O'Toole | Fall 2024 |
| [16-385 Computer Vision — Lectures](https://16385.courses.cs.cmu.edu/spring2026/lectures) | CMU | — | Spring 2026 |
| [CS194-26/294-26 Intro to Computer Vision and Computational Photography](https://szeliski.org/Book/) | Berkeley | Alyosha Efros | Fall 2024 |
| [15-463, 15-663, 15-862 Computational Photography](https://graphics.cs.cmu.edu/courses/15-463/) | CMU | Ioannis Gkioulekas | Fall 2024 |
| [CSCI 1430 Computer Vision](https://szeliski.org/Book/) | Brown | James Tompkin | Spring 2025 |
| [CMPT 412 and 762 Computer Vision](https://szeliski.org/Book/) | Simon Fraser | Yasutaka Furukawa | Fall 2023 |
| [CS 4476-A / 6476-A Computer Vision](https://szeliski.org/Book/) | Georgia Tech | James Hays | Fall 2022 |
| [EECS 498.008 / 598.008 Deep Learning for Computer Vision](https://szeliski.org/Book/) | U Michigan | Justin Johnson | Winter 2022 — *outstanding intro to deep learning and visual recognition* |
| [DS-GA 1008 Deep Learning](https://szeliski.org/Book/) | NYU | Yann LeCun, Alfredo Canziani | Spring 2021 |
| [Fundamentals and Trends in Vision and Image Processing](https://szeliski.org/Book/) | IMPA | Luiz Velho | Spring 2021 |
| [CS294-158 Deep Unsupervised Learning](https://szeliski.org/Book/) | UC Berkeley | — | Spring 2020 |
| [CSCI 497P/597P Introduction to Computer Vision](https://szeliski.org/Book/) | Western Washington | Scott Wehrwein | Spring 2020 |
| [EECS 504 Foundations of Computer Vision](https://szeliski.org/Book/) | U Michigan | Andrew Owens | Winter 2020 |

*Course links are maintained at [szeliski.org/Book](https://szeliski.org/Book/). Contact the author to add your course.*

---

## Links

| Resource | URL |
|----------|-----|
| 07-280 AI & ML I | https://www.cs.cmu.edu/~07280/#schedule |
| 16-385 Computer Vision Spring 2026 Lectures | https://16385.courses.cs.cmu.edu/spring2026/lectures |
| 15-463 Computational Photography | https://graphics.cs.cmu.edu/courses/15-463/2018_fall/ |
| **Szeliski: Computer Vision (2nd ed.)** | https://szeliski.org/Book/ |

---

*Last updated: February 2025. Course offerings and requirements may change; verify with CMU official sources.*
