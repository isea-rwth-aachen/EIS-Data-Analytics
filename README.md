# EIS Data Analytics

<a href="https://www.carl.rwth-aachen.de/?lidx=1" target="_blank">
    <img src="misc/CARL_ISEA_Logo.svg" align="right" alt="CARL Logo"  height="80"/>
</a>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*EIS Data Analytics* is a collection of code to analyse which fitting approach is suitable for your electrochemical impedance spectroscopy (EIS) data.  
<a href="https://doi.org/10.1016/j.jpowsour.2024.235049" target="_blank">
<img src="misc/graphical_abstract.svg" alt="Graphical Abstract [1]" height="250"/> [1]
</a>  

- [Getting Started](README.md#getting-started)
- [Example Usage](README.md#example-usage)
- [Overview](README.md#overview)
- [Temperature Estimation](README.md#temperature-estimation)
- [Verification](README.md#verification)
- [Colophon](README.md#colophon)
- [Further Information](README.md#further-information)
- [FAQ](README.md#faq)
- [Sources](README.md#sources)


# Getting Started
1.  Clone the project into your "path_to_this_project". Clone the submodules is important here:
	```bash
    git clone --recurse-submodules https://git.rwth-aachen.de/isea/eis_data_analytics.git
    ```
2. Install Python 3.11.9 (https://www.python.org/downloads/)
3. Open an prompt and navigate to the path of this project
    ```bash
    cd path_to_this_project
    ```
4. Follow the instructions of the [Python Read the Docs](https://docs.python.org/3.11/library/venv.html) to create an virtual environment (venv) and activate it. E.g.:  
	**Windows**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
    **Linux / Mac**
    ```bash
    python -m venv .venv
    . .venv/bin/activate
    ```

5. Install all necessary packages with:  
    **Windows**
	```bash
    pip install -r requirements_windows.txt
    ```
    **Linux / Mac**
	```bash
    pip install -r requirements.txt
    ```
6. (Optional) Clean up your venv:
	```bash
    python -m pip cache purge
    ```
7. Open this project with an Jupyter-Notebook editor of your choice, e.g. [VS Code](https://code.visualstudio.com/Download) (needs to be installed separatly) with:
    ```bash
    code
    ```

# Example Usage
Details and step-by-step explanations can be found in the Jupyter Notebooks:
1. [a_data_preparation.ipynb](a_data_preparation.ipynb)
2. [b_data_inspection.ipynb](b_data_inspection.ipynb)
3. [c_input_data_definition.ipynb](c_input_data_definition.ipynb)
4. [d_regression_analysis_individual_frequencies.ipynb](d_regression_analysis_individual_frequencies.ipynb)
5. [e_regression_analysis_frequency_spectrum.ipynb](e_regression_analysis_frequency_spectrum.ipynb)


# Overview
The main purpose of this tool is to process and analyze large amounts of EIS measurements. This implies that compromises have to be made in some respects. The following functions are already implemented:
- Standardization of the frequency measuring points by interpolation
- LinKK
- ECM fitting
- DRT fitting
- Impedance extrema extraction
- Various forms of visualization
- Support Vector Regression (SVR)

Will follow soon:
- Phase estimation based on absolute values
- Prediction based on machine learning (ML)

# Temperature Estimation
A more detailed explanation can be found in the corresponding publication. [1]  
The cells “LiFun_575166-01_XXX.csv” in the example_input_data folder are used to develop an example SVR temperature estimation model. The data includes the temperatures -15, -5, 5, 15, 25, 35, 45 and 55 °C. An EIS measurement is carried out at each temperature and the SOC values 0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99 and 100 %.  
<a href="https://doi.org/10.1016/j.jpowsour.2024.235049" target="_blank">
<img src="misc/example_data_bode.svg" alt="Example Data Bode [1]" height="450"/> [1]
</a>  
The following results are obtained for three individual frequencies.  
<a href="https://doi.org/10.1016/j.jpowsour.2024.235049" target="_blank">
<img src="misc/example_data_svr_single_f_temperature.svg" alt="EIS Temperature Estimation SVR, Input: one frequency [1]" height="400"/> [1]
</a>    
With the entire spectrum as input, the following errors can be obtained.  
<a href="https://doi.org/10.1016/j.jpowsour.2024.235049" target="_blank">
<img src="misc/example_data_svr_all_fs_temperature.svg" alt="EIS Temperature Estimation SVR, Input: all frequencies [1]" height="200"/> [1]
</a>  

# Verification
To verify the flexibility of the tool, the results of the publication "Accelerated state of health estimation of second life lithium-ion batteries via electrochemical impedance spectroscopy tests and machine learning techniques" by Mona Faraji-Niri et al. (https://doi.org/10.1016/j.est.2022.106295) on their dataset by Muhammad Rashid et al. (https://doi.org/10.1016/j.dib.2023.109157) is verified. Different to their paper we use the SVR to estimate the SOH. We used three different input combinations. For each 30.000 hyperparameter combinations were tested. Below for each input combination the best result.

#### SOH prediction based on Absolute Values of the Impedance
<img src="misc/reference_data_output_soh_input_abs_result.svg" alt="SOH prediction based on Absolute Values of the Impedance Result" height="300"/>
<img src="misc/reference_data_output_soh_input_abs_parameter.svg" alt="SOH prediction based on Absolute Values of the Impedance Hyperparameter" height="300"/>

#### SOH prediction based on Absolute Values of the Impedance + Temperature + SOC
<img src="misc/reference_data_output_soh_input_abs_temp_soc_result.svg" alt="SOH prediction based on Absolute Values of the Impedance Result" height="300"/>
<img src="misc/reference_data_output_soh_input_abs_temp_soc_parameter.svg" alt="SOH prediction based on Absolute Values of the Impedance Hyperparameter" height="300"/>

#### SOH prediction based on Absolute and Phase Values of the Impedance + Temperature + SOC
<img src="misc/reference_data_output_soh_input_abs_phase_temp_soc_result.svg" alt="SOH prediction based on Absolute Values of the Impedance Result" height="300"/>
<img src="misc/reference_data_output_soh_input_abs_phase_temp_soc_parameter.svg" alt="SOH prediction based on Absolute Values of the Impedance Hyperparameter" height="300"/>

# Colophon

Thanks to Spyridon, Xiaohan and Jacob who supported us, <a href="https://orcid.org/0000-0001-8394-5859">Ole Kappelhoff <img alt="ORCID Logo" src="misc/ORCIDiD_iconvector.svg" width="16" height="16" /></a> and <a href="https://orcid.org/0000-0003-0943-9485">Alexander Blömeke <img alt="ORCID Logo" src="misc/ORCIDiD_iconvector.svg" width="16" height="16" /></a>.

We are using [impedance.py](https://github.com/ECSHackWeek/impedance.py) for LinKK calculations and RC fitting: https://doi.org/10.21105/joss.02349.
For DRT calculations we use [pyDRTtools](https://github.com/ciuccislab/pyDRTtools): https://doi.org/10.1016/j.electacta.2019.135316.

## Related Publications / Citation

Please cite our paper: https://doi.org/10.1016/j.jpowsour.2024.235049  

Archived versions of this git:  
Release v0.0.9: https://doi.org/10.18154/RWTH-2024-03849

## License

This project is licensed according to the file [LICENSE](/LICENSE "LICENSE").

## Acknowledgement

The authors acknowledge the financial support by the Federal Ministry of Education and Research (BMBF) of Germany in the project OSLiB (project number 03XP0330C) within the competence cluster Battery Utilisation Concepts (BattNutzung).


<a href="https://www.bmbf.de/bmbf/en" target="_blank">
    <img src="misc/BMBF_Logo.svg" alt="BMBF Logo" height="120"/>
</a>
<a href="https://www.battnutzung-cluster.de/en/projects/oslib/" target="_blank">
    <img src="misc/OSLiB_Logo.svg" alt="OSLiB Logo" height="120"/>
</a>


# Further Information



## Developer Info

We use Git Large File Storage (LFS).

This project is written in [Python 3.11.9](https://www.python.org/) using [Visual Studio Code](https://code.visualstudio.com/) and [Jupyter Notebooks](https://jupyter.org/) in an Python virtual environment on Windows.
A [requirements.txt](requirements.txt) can be created by:

```bash
pip freeze -l > requirements.txt
```

## Git and Jupyter Notebooks
Consider to ignore the Jupyter Outputs in Git:

```bash
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
```



# FAQ
- **pywin32 fails**: https://github.com/microsoft/vscode-jupyter/wiki/Failure-to-start-kernel-due-to-failures-related-to-win32api-module
- **general_fun_v8 can't be found**: Make sure you loaded the submodule. See step 1. at [Getting Started](README.md#getting-started)
- **I want to run the hyperparameter search in parallel**: It is recommended to use linux/wsl for this. Make sure you install java.


# Sources
[1] https://doi.org/10.1016/j.jpowsour.2024.235049