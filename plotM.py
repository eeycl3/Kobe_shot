#n_estimators:  17 max_depth:  8 score:  0.678031683122696 std:  0.004661720399105917

import matplotlib.pyplot as plt
"""
matrix = [[0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585], [0, 0.6139058150008706, 0.6258715128175556, 0.6385923086670134, 0.6598510761278141, 0.6599091886230372, 0.6606061640382226, 0.6655440154525588, 0.6778574266159438, 0.6777993343654501, 0.6777993343654501, 0.6776250879810627, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585, 0.6772185130841585]]
plt.title("20 features random forest parameters vs accuracy scores")
plt.imshow(matrix, cmap="hot", vmax=0.7, vmin=0.6)
plt.xlabel("number of trees")
plt.ylabel("number of max depth")
ax = plt.colorbar()
ax.set_label("accuracy score")
plt.show()
"""
matrix = [[0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005], [0, 0.6134411579758373, 0.6152998164430646, 0.6439357733536162, 0.6552041819618126, 0.6604319783880227, 0.6625812803102821, 0.6788447620599523, 0.6775668742621935, 0.6771020755240557, 0.6765794780839975, 0.6757082259173307, 0.6754758366706263, 0.67170036675149, 0.6713518638603505, 0.6700739254507685, 0.6674601487060388, 0.6676343849680618, 0.6663566287610431, 0.6653691414815656, 0.665369131359201, 0.6646140130816987, 0.6648463922060385, 0.6645559613206633, 0.6644978893148986, 0.6643236530528758, 0.664381735181005, 0.664381735181005, 0.664381735181005, 0.664381735181005]]

plt.title("30 features random forest parameters vs accuracy scores")
plt.imshow(matrix, cmap="hot", vmax=0.7, vmin=0.6)
plt.xlabel("number of trees")
plt.ylabel("number of max depth")
ax = plt.colorbar()
ax.set_label("accuracy score")
plt.show()