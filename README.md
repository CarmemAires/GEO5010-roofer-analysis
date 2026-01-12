# GEO5010 Roofer Analysis 

This repository contains a set of Python scripts to process LiDAR ```LAZ``` tiles into 3D CityJSON models using **Roofer** (https://github.com/3DBAG/roofer), analyze their quality, and optionally optimize Roofer parameters for improved reconstruction. These scripts were developed as part of GEO5010 – Research Orientation in the MSc Geomatics program at TU Delft, with the aim of testing the Roofer algorithm in Global South contexts, such as São Paulo.

## Overview of Scripts

1. ```process_roofer_parallel_tuned.py```: Downloads ```.LAZ``` tiles from ```BR17_SaoPaulo_TileIndex.gpkg``` which contains a Typology field, crops building footprints, runs Roofer in parallel to generate 3D roof models, and converts Roofer outputs (```.city.jsonl```) to standard CityJSON (```.city.json```) using CJIO.

    ### inputs
    - Tile index GeoPackage (GPKG_PATH) with Typology column

    -  Master building footprints GeoPackage (FOOTPRINTS_GPKG)

    - Roofer executable (ROOFER_EXE)

    - LAZ tiles (URLs specified in the GeoPackage)

    ### outputs
    

    - ```.city.jsonl``` and ```.city.json``` files for each tile
    - cropped building footprints in ```.GPKG``` for each tile

    ### configuration example
    ```
    GPKG_PATH = Path(r"data/BR17_SaoPaulo_TileIndex.gpkg")
    DEST_DIR = Path(r"data/output")
    ROOFER_EXE = Path(r"bin/roofer.exe")
    FOOTPRINTS_GPKG = DEST_DIR / "footprints.gpkg"
    MAX_WORKERS = 3
    ROOFER_PARAMS = [
        "--complexity-factor", "0.741",
        "--plane-detect-min-points", "21",
        "--plane-detect-epsilon", "0.131",
        "--plane-detect-k", "20",
    ]
    ```

2. ```analyze_cityjson_quality.py```: Performs detailed quality analysis of CityJSON outputs per typology, generating metrics such as RMSE, val3dity validation codes, extrusion success, and overlap with footprints. Produces visualizations and summary reports.

    ### inputs
    - CityJSON files (```.city.json```) and corresponding footprints ```.GPKG``` produced by process_roofer_parallel_tuned.py

    ### outputs
    

    - aggregated_statistics_by_typology.json

    - summary_statistics.csv

    - Typology-specific plots (PNG)

    - Updated GPKGs with quality indicators

    
    ### configuration example
    ```
    DEST_DIR = Path(r"data/output")
    FOOTPRINTS_BASE_NAME = "footprints"
    ```

3. ```optimize_roofer_params.py```: **(Optional)** Performs Bayesian hyperparameter optimization for Roofer reconstruction parameters using Optuna (TPE). Evaluates each parameter set against CityJSON outputs and maximizes reconstruction quality.

    ### inputs
    - Working directory with LAZ tiles, footprints, and Roofer outputs
    - Roofer executable (ROOFER_EXE)

    ### outputs
    
    - Optimized Roofer parameters for improved reconstruction

    
    ### configuration example
    ```
    WORK_DIR = Path(r"data/output")
    ROOFER_EXE = Path(r"bin/roofer.exe")
    ```


## Requirements

1. Python >= 3.9
2. Dependencies: use ```src/requirements.txt``` to install the dependencies,
        ```          
        geopandas, pandas, numpy, shapely, tqdm, matplotlib, seaborn, requests, optuna
        ```
3. External Tools:
    - Roofer executable accessible ```(ROOFER_EXE)```
    - CJIO 

4. Input Data: 
    - tile index ```.GPKG``` file with a ```URL``` field to point-cloud data download.
    - Building footprint with unique ```ID``` field.
    - Data should be in the same ```CRS```

## Recommended Workflow

1. Download the tile index in https://portal.opentopography.org/lidarDataset?opentopoID=OTLAS.062020.31983.1. 
Add your typologies in a ```Typology``` field in the tile index dataset. ```Typology``` allows you to analyze results grouped by building type. Tip: If your dataset has many tiles of the same type, you can assign the same typology to all of them (e.g. "Residential"), so Roofer and analysis scripts will group them automatically.
    - Create a column called Typology in your tile index  ```.GPKG``` (e.g. "BR17_SaoPaulo_TileIndex.gpkg").
    - Use meaningful labels per tile type. 
    - Null or missing typology tiles will be skipped automatically.
    
    Download footprint dataset(s), for example: ```Edificações 2D``` in https://geosampa.prefeitura.sp.gov.br/PaginasPublicas/_SBC.aspx

2. Download tiles, generate 3D models, convert to CityJSON.
    ```
    python3 process_roofer_parallel_tuned.py
    ```

3. Generate statistics, visualizations, and quality metrics per typology.
    ```
    python3 analyze_cityjson_quality.py
    ```

4. Optional: Optimize Roofer parameters: Tune Roofer parameters for better reconstruction quality on your dataset.
    ```
    python3 optimize_roofer_params.py
    ```
