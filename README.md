### Running the map
1. Place build_population_map.py and map_ui.py in the same folder as your data:

  - population.pkl
  - all_hospitals.pkl
  - distances_osm_max_300km.pkl
  - road_osm_preprocessed.geojson
  - stroke-facs-100-en.csv

2. Change the folder paths in the file to your folder path, around line 40.

3. Run the build_population_map.py file.

4. Open the generated population_map.html in your browser.

### Configuration
You can change the size of the sample of the population points being displayed. Beware the entire population point dataset will likely crash your browser or run for multiple hours.

There are two ways to change that:
- Adjust sample size (default 1000 interactive population points):
- Restrict by latitude (for faster testing on a subregion):
either in this function definition: 
<img width="514" height="274" alt="image" src="https://github.com/user-attachments/assets/81f82639-d241-4b26-bc6f-8e966e1550c2" />

or in the function call at the end of the file: 
<img width="395" height="178" alt="image" src="https://github.com/user-attachments/assets/f63e9ae5-d80b-4ace-8aa6-26ab058fb48c" />

