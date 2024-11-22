# EIS Data Analytics - Microcontroller EIS Network Code

<a href="https://www.carl.rwth-aachen.de/?lidx=1" target="_blank">
    <img src="../misc/CARL_ISEA_Logo.svg" align="right" alt="CARL Logo"  height="80"/>
</a>

Getting started with the microcontroller code.

# Getting Started

1. Open the project in the STM32CubeIDE by clicking on the **".cproject"** file. Or open STM32CubeIDE and File->Import->General->Existing Project into Workspace->Browse->"this folder".
2. Open the microcontroller_eis_network->**microcontroller_eis_network.ioc** file of the project by clicking on it.
3. Click on X-CUBE-AI at "Pinout & Configuration" -> "Middleware and Software Packages"  
<img src="../misc/microcontroller_1.png" height="500"/>

4. Update the eis_network by selecting an ONNX model from the folder onnx_export. DON'T rename anything. Otherwise the rest of the code isn't working anymore.  
<img src="../misc/microcontroller_2.png" height="200"/>

5. Analyze the network.  
<img src="../misc/microcontroller_3.png" height="265"/>

6. Close the .ioc file and say yes to save changes. The code should now be updated automatically. The first time might take wile.  
<img src="../misc/microcontroller_4.png" height="50"/>  
<img src="../misc/microcontroller_5.png" height="150"/>

7. Open the microcontroller_eis_network->Core->Src->main.c and build the project. If something is wrong check again the main [README.md](../README.md) to see if some requirements are not met.
8. Flash the code and see the results send by UART.  
<img src="../misc/microcontroller_6.png" height="150"/>  

# Update the Model
Repeat the steps from “Getting started”, but select your new ONNX model. The header should be automatically updated by the corresponding Python code.

# Related Publications / Citation

Please cite our paper: https://doi.org/10.1016/j.jpowsour.2024.235049 .  

Archived versions of this git:  
Release v0.0.9: https://doi.org/10.18154/RWTH-2024-03849  .