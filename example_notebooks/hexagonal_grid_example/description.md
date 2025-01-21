# Example description

This example is a series of notebooks that break out calculating access to grocery stores for a hexagonal grid that covers Toronto.

1. `1--create_hexagonal_grid.ipynb`: Shows how to use the `accessopp.point.build_haxagonal_grid` function to build a hexagonal grid covering Toronto,
is trimmed at the boundaries to follow Toronto's boundaries. 
2. `2--read_DineSafe_opendata.ipynb`: Provides an example of how to read DineDame open data in using pandas, and then save as a shapefile for future processing.
3. `3--calculate_access_to_grocery_stores.ipynb`: Calculate travel times to third closest supermarket for each hexagon created in part 1.
