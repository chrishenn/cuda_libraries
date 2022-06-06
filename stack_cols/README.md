# Stack Columns
### stack_col
###

Provides the same transformation as write_row, but supports an arbitrary and limited output data size.

Vectors that occupy a tensor-row are written contiguously into wider rows of an output structure. We can specify some arbitrary size of this output vector to accomodate some number of features.