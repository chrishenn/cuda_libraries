# Grid Bin, One Bin, Ring Bin, Score Bin
### grid_bin, oned_bin, ring_bin, score_bin
###

These extensions each provide some comparison of objects within predefined segmentations ('bins'), according to some score. They will judge objects that pass or fail within each image, and provide the information necessary to locate the passing or failing objects. 

They each follow a similar programming pattern, and provide slightly different functionality. It is more efficient (for prototyping) to compile each extension separately, rather than attempt to template these operations into one codebase.