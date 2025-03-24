# GLSLR-DEM

This repository contains all the scripts and most of configuration files needed to create 1m resolution topo-bathymetric DEM in
Lake Ontario and the fluvial stretch of St. Lawrence River floodplain.

It was created from 2022 to 2025, for the GLAM expedited review of Plan 2014.

The following data are not provided within this repo:

	- Bathymetric and topographic elevation source data (/DATA)
	- NDVI mosaic from Sentinel 2 images (/correction_model/NDVI)
	- Bathymetric dataset extent shapefiles	(/bathy)
	- Bathymetric dataset extent shapefiles	(/topo)
	- The extent shapefiles of source datasets used to create topobathy DEM in the United States portion of the AOI. (/USACE_metadata)
	- Areas where elevation data are missing or are of unsuficient quality to be integreatd in the final DEM (/data_gaps)
	
Hyperlink towards publicaly availbale source data can be found in list_main_datasets_details.csv 

To request access to any other files, please contact National Hydrological Services - Hydrodynamic Ecohydraulic Section:

Antoine Maranda (antoine.maranda@ec.gc.ca); 
Dominic Theriault (dominic.theriault@ec.gc.ca);
Patrice Fortin (patrice.fortin@ec.gc.ca);
Marianne Bachand(Marianne.bachand.ec.gc.ca);

Conda environment with all necessary dependencies, can be created with GLSLR_DEM.yml file.
Note that the following softwares need to be downloaded and installed locally:
- WhiteboxTools Open Core from Whotebox geospatial, has to be downloaded and installed locally. (https://www.whiteboxgeo.com/download-whiteboxtools/)
- GDAL (https://gdal.org/en/stable/download.html)

![Preview](https://raw.githubusercontent.com/eccc-Antoine/GLSLR-DEM/main/docs/assets/images/GLSLR_DEM_Workflow.png)

Author: Antoine Maranda (ECCC, NHS-HES)
