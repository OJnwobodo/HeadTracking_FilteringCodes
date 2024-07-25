# HeadTracking_FilteringCode

## Overview
This repository contains the code for the adaptive Kalman-particle filter fusion strategy for head tracking in AR flight simulators using Microsoft HoloLens 2. This project aims to enhance accuracy and reduce head tracking latency by leveraging the complementary strengths of Kalman and particle filters, which are optimized for AR flight simulation requirements.


## Introduction
The adaptive Kalman-particle filter (AKPF) method is designed to improve head-tracking performance in augmented reality (AR) flight simulators. The approach integrates Microsoft HoloLens 2 with a flight simulation environment to provide accurate and responsive head tracking.

## Installation
To run the code, you will need to have Unity installed along with the necessary dependencies(MRTK) for HoloLens development. Follow these steps to set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/OJnwobodo/HeadTracking_FilteringCode.git
2. Open the project in Unity:

Launch Unity Hub.
Click on "Add" and navigate to the cloned repository folder.
Select the folder to add the project to Unity Hub and open it.
3. Install the required packages:

 Open the Package Manager (Window -> Package Manager).
Ensure you have installed the necessary packages such as XR Plugin Management, Mixed Reality Toolkit (MRTK), and any other required dependencies.
Usage
Once the environment is set up, you can run the head-tracking algorithm by following these steps:

Connect your Microsoft HoloLens 2 to your development machine.
Open the appropriate scene in Unity that contains the head tracking setup.
Build and run the project on the HoloLens 2:
Go to File -> Build Settings.
Select the Universal Windows Platform (UWP) and configure the build settings as required.
Click "Build and Run" to deploy the application to HoloLens 2.
Code Structure
Assets/Scripts/main.cs: The main script to run the head tracking algorithm.
Assets/Scripts/kalman_filter.cs: Contains the implementation of the Kalman filter.
Assets/Scripts/particle_filter.cs: Contains the particle filter implementation.
Assets/Scripts/akpf.cs: Implements the adaptive Kalman-particle filter fusion strategy.
Experimental Setup
The experimental setup involves integrating Microsoft HoloLens 2 with a flight simulation environment to evaluate the performance of the AKPF method. Detailed instructions for setting up the simulation environment and connecting HoloLens 2 are provided in the setup_instructions.md file.

Contributing
We welcome contributions to enhance the project. Please fork the repository, create a new branch, and submit a pull request with your changes. Ensure your code adheres to the project's coding standards and includes appropriate documentation.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any questions or inquiries, please contact [onyeka.nwobodo@polsl.pl].


This README file reflects that the code is written in C# and developed in Unity, with specific instructions for setting up and running the project in that environment. Ensure that the file names and paths correspond to your actual project structure.

