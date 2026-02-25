# Carnegie Mellon University: Robotics & AI Courses Reference

A comprehensive reference of CMU's School of Computer Science (SCS) undergraduate programs in **Artificial Intelligence** and **Robotics**, compiled from official course pages and catalogs.

**Sources:**
- [07-280 AI & ML I](https://www.cs.cmu.edu/~07280/#schedule)
- [15-463 Computational Photography](https://graphics.cs.cmu.edu/courses/15-463/2018_fall/)
- [Bachelor of Science in Robotics](https://www.ri.cmu.edu/education/academic-programs/bachelor-of-science-in-robotics/)
- [Robotics Program Catalog](http://coursecatalog.web.cmu.edu/schools-colleges/schoolofcomputerscience/robotics/#roboticscurriculumtextcontainer)

---

## Table of Contents

1. [07-280: AI & Machine Learning I](#1-07-280-ai--machine-learning-i)
2. [15-463: Computational Photography — Deep Insight Before Computer Vision](#2-15-463-computational-photography--deep-insight-before-computer-vision)
3. [Bachelor of Science in Robotics (BSR)](#3-bachelor-of-science-in-robotics-bsr)
4. [Robotics Minor & Additional Major](#4-robotics-minor--additional-major)
5. [Key Robotics Courses (Catalog)](#5-key-robotics-courses-catalog)
6. [Self-Study Mapping](#6-self-study-mapping-to-roadmap)
7. [Additional Resources: Szeliski Book & Related Courses](#7-additional-resources-szeliski-book--related-courses)

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
| **Textbook** | No required textbook; readings from AIMA, Bishop, Daumé, Goodfellow, MML, Mitchell, Murphy, KMPA (all online or via CMU Library) |

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

**Grade boundaries (rough):** A ≥90%, B 80–90%, C 70–80%, D 60–70%. Not curved.

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
- **Participation:** ≥80% of in-class polls for full credit
- **Collaboration:** Conceptual discussion allowed; no sharing code/text; generative AI may not be used to generate submissions
- **Programming partners:** Groups of 2 allowed for programming components only

### Comparison: 07-280 vs 10-301

| 07-280 | 10-301 |
|--------|--------|
| Heuristic Search, Adversarial Search, CSPs | — |
| ML Parallelism/GPU Basics | — |
| Monte Carlo Tree Search | — |
| Transformer networks, LLMs | ✓ |
| Reinforcement Learning | ✓ |
| ML fundamentals (decision trees → neural nets) | ✓ |
| Fulfills AI Major core | ✓ |
| Prereq for 07-380 AI & ML II | ✓ |

---

## 2. 15-463: Computational Photography — Deep Insight Before Computer Vision

**Recommended before 16-385 Computer Vision.** Provides foundational understanding of imaging physics, camera pipelines, and computational methods that underpin both graphics and vision.

*Source: [15-463 Fall 2018](https://graphics.cs.cmu.edu/courses/15-463/2018_fall/)*

### Overview

| Item | Details |
|------|---------|
| **Cross-listing** | 15-463 (undergrad), 15-663 (Master's), 15-862 (PhD) |
| **Schedule** | Mon + Wed, 12:00–1:20 PM |
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

## 3. Bachelor of Science in Robotics (BSR)

**Housed in SCS.** Announced Fall 2023. Emphasizes problem-solving, systems thinking, and interdisciplinary teams.

### Core Concepts

- **RoboMath** — Mathematical foundations for robotics
- **Computer Vision and Sensing**
- **Mechanisms and Manipulation**
- **General Robotic Systems**
- **Planning and Control**
- **Robotic Building Practices**
- **Capstone Project**

### Degree Requirements Summary

| Area | Courses | Units |
|------|---------|-------|
| Robotics (core + electives) | 11 | 125 |
| Computer Science (core + SCS elective) | 6 | 51 |
| Mathematics | 6 | 67 |
| Ethics | 1 | 12 |
| Science / Engineering | 4 | 36 |
| Humanities / Arts | 7 | 63 |
| Computing @ CMU | 1 | 3 |
| First Year Seminar | 1 | 3 |
| **Total** | **26** | **360** |

### Computer Science Core

| Course | Name |
|--------|------|
| 07-128 | First Year Immigration Course |
| 15-122 | Principles of Imperative Computation |
| 15-213 | Introduction to Computer Systems |
| 15-251 | Great Ideas in Theoretical Computer Science |

### Robotics Core

| Course | Name |
|--------|------|
| 16-280 or 16-311 | Intelligent Robot Systems (or Intro to Robotics; 16-281 General Robotics starting Spring 2026) |
| 16-220 | Robot Building Practices |
| 16-299 | Introduction to Feedback Control Systems |
| 16-384 | Robot Kinematics and Dynamics |
| 16-385 | Computer Vision |
| 16-450 | Robotics Systems Engineering (senior fall) |
| 16-474 | Robotics Capstone (senior spring) |

### Ethics (one of)

- 16-161 Artificial Intelligence and Humanity
- 16-735 Ethics and Robotics

### Mathematics

| Course | Name |
|--------|------|
| 15-151 | Mathematical Foundations for Computer Science |
| 16-211 | Foundational Mathematics of Robotics |
| 21-120 | Differential and Integral Calculus |
| 21-122 | Integration and Approximation |
| 21-241 | Matrices and Linear Transformations |
| + Probability (15-259, 21-235, 36-218, or 36-225) | |

### Robotics Electives

3 general robotics electives (16-3xx, 16-4xx). Up to 12 units of 16-597 (Undergraduate Reading and Research) or 99-270 (Summer Undergraduate Research) may count.

### Sample Course Sequence (4 years)

| Year | Fall | Spring |
|------|------|--------|
| **1** | 07-128, 15-122, 15-151, 21-122, 76-101, 99-101 | 15-213, 16-180, 21-241, Science/Eng, H&A |
| **2** | Probability, 16-220, 16-211, H&A | 16-299, 15-251, 16-385, H&A |
| **3** | 16-384, H&A, 16-280, H&A, Free | Robotics Elective ×2, H&A, Free |
| **4** | 16-450, Ethics, Science/Eng, Robotics Elective | 16-474, SCS Elective, Science/Eng, Free |

### Notable Courses

- **16-281 General Robotics** — Rube Goldberg machines, mini Urban Search and Rescue (replaces 16-311 from Spring 2026)
- **16-384 Robot Kinematics and Dynamics** — Hebi robotic arm control
- **16-385 Computer Vision** — Image processing, detection, recognition, geometry-based vision

---

## 4. Robotics Minor & Additional Major

### Robotics Minor

- **Overview:** 16-280 or 16-311 (16-281 from Spring 2026)
- **Controls:** One of 06-464, 16-299, 18-370, 24-451, 24-773
- **Mechanisms:** 16-384 or (15-362 + 33-141)
- **Robot Building:** One of 16-220, 16-362, 16-423, 18-349, 18-500, 18-578, 24-671
- **Electives:** 2 general robotics electives
- **Double-count:** Max 2 courses from major
- **QPA:** 2.5 in minor curriculum

### Additional Major in Robotics

- **Application:** Jan 7–Feb 3
- **Prereq:** C language, basic programming, algorithms (15-122)
- **Double-count:** Max 6 courses from primary major (5 for CS majors)
- **QPA:** 3.0 in additional major

---

## 5. Key Robotics Courses (Catalog)

| Code | Name | Notes |
|------|------|-------|
| 16-180 | Concepts of Robotics | Intro; perception, cognition, action |
| 16-211 | Foundational Mathematics of Robotics | Coordinate transforms, Jacobians, optimization, neural nets |
| 16-220 | Robot Building Practices | CAD, 3D printing, electronics, PCB, motor controllers |
| 16-280 | Intelligent Robot Systems | Search, planning, perception, control, ROS, embedded |
| 16-299 | Introduction to Feedback Control Systems | Classical control, state-space, LQR, nonlinear |
| 16-311 | Introduction to Robotics | LEGO robots, vision, motion planning, kinematics (→ 16-281) |
| 16-350 | Planning Techniques for Robotics | Path/motion planning, ground/aerial/humanoids |
| 16-362 | Mobile Robot Algorithms Laboratory | Multirotor autonomy, Python/C++ simulator |
| 16-384 | Robot Kinematics and Dynamics | Forward/inverse kinematics, Jacobians |
| 16-385 | Computer Vision | Image processing, detection, recognition |
| 16-450 | Robotics Systems Engineering | Systems engineering, design, prototyping |
| 16-474 | Robotics Capstone | Build, integrate, test robot from 16-450 |
| 16-663 | F1Tenth Autonomous Racing | 1/10 scale autonomous race car |
| 16-664 | Self-Driving Cars: Perception & Control | Deep learning, sensor fusion, localization |
| 16-735 | Ethics and Robotics | AI/ML in society, power, ethics |
| 16-831 | Introduction to Robot Learning | RL, imitation learning, visual learning |
| 16-884 | Deep Learning for Robotics | Robot learning, deep RL, control |

### Graphics & Imaging (Before Computer Vision)

| Code | Name | Notes |
|------|------|-------|
| **15-462** | Computer Graphics | Rendering, transforms, shading — foundation for 15-463 |
| **15-463** | Computational Photography | Imaging physics, camera pipelines, HDR, lightfields — **deep insight before 16-385** |

---

## 6. Self-Study Mapping to Roadmap

For learners following the **AI Hardware Engineer Roadmap**, here is how CMU’s curriculum aligns:

| Roadmap Phase | CMU Course(s) | Overlap |
|---------------|---------------|---------|
| **Phase 3: Computer Vision** | **15-463** (Computational Photography) → 16-385 (Computer Vision) | Imaging physics, camera pipelines, then detection/recognition |
| **Phase 4: Jetson Edge AI** | 07-280 (Neural Nets, AlexNet, PyTorch), 16-385 (Computer Vision) | Neural networks, computer vision, edge deployment |
| **Phase 4: Sensor Fusion** | 16-299 (Control), 15-463 (imaging), 16-385 (Vision) | Kalman filtering, sensor fusion concepts |
| **Phase 4: ROS2** | 16-280 (ROS), 16-220 (Robot Building) | ROS, embedded systems, robot software |
| **Phase 5: Robotics** | 16-220, 16-280, 16-384, 16-450, 16-474 | Full robotics stack: mechanics, control, planning |
| **Phase 5: Autonomous Driving** | 16-663 (F1Tenth), 16-664 (Self-Driving Cars) | Perception, planning, control for autonomy |
| **Phase 5: AI Chip Design** | 07-280 (optimization, GPU/parallel), 16-211 (math) | Optimization, linear algebra, parallel compute |

### Suggested Self-Study Order (CMU-Inspired)

1. **15-122 equivalent** — Imperative programming (C/Python)
2. **21-120, 21-122, 21-241** — Calculus, linear algebra
3. **07-280 topics** — Search → ML → Neural Nets → RL → Transformers
4. **16-220** — Robot building (CAD, electronics, prototyping)
5. **16-299** — Feedback control
6. **16-384** — Kinematics and dynamics
7. **15-462** (optional) — Computer Graphics — rendering, transforms
8. **15-463** — **Computational Photography** — imaging physics, camera pipelines, HDR, lightfields *(deep insight before vision)*
9. **16-385** — Computer vision
10. **16-280** — Intelligent robot systems (ROS, planning, perception)

---

## 7. Additional Resources: Szeliski Book & Related Courses

### Computer Vision: Algorithms and Applications (2nd ed.)

**[https://szeliski.org/Book/](https://szeliski.org/Book/)** — Richard Szeliski, University of Washington (© 2022)

The canonical computer vision textbook. Free PDF download for personal use. Used by 15-463 Computational Photography and many vision courses worldwide. Covers image formation, feature detection, stereo, structure from motion, recognition, and more.

### Related Courses (from [Szeliski Book](https://szeliski.org/Book/))

Additional good sources for computer vision and computational photography, sorted roughly by most recent:

| Course | Institution | Instructor(s) | Term |
|--------|-------------|---------------|------|
| [CS5670 Introduction to Computer Vision](https://www.cs.cornell.edu/courses/cs5670/2025sp/) | Cornell Tech | Noah Snavely | Spring 2025 |
| [6.8300/6.8301 Advances in Computer Vision](https://szeliski.org/Book/) | MIT | Bill Freeman, Antonio Torralba, Phillip Isola | Spring 2023 |
| [16-385 Computer Vision](http://www.cs.cmu.edu/~16385/) | CMU | Matthew O'Toole | Fall 2024 |
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
| 15-463 Computational Photography | https://graphics.cs.cmu.edu/courses/15-463/2018_fall/ |
| B.S. Robotics | https://www.ri.cmu.edu/education/academic-programs/bachelor-of-science-in-robotics/ |
| Robotics Catalog | http://coursecatalog.web.cmu.edu/schools-colleges/schoolofcomputerscience/robotics/ |
| SCS Undergraduate | https://www.cs.cmu.edu/academics/undergraduate/ |
| CMU Admission | https://admission.cmu.edu/ |
| **Szeliski: Computer Vision (2nd ed.)** | https://szeliski.org/Book/ |

---

*Last updated: February 2025. Course offerings and requirements may change; verify with CMU official sources.*
