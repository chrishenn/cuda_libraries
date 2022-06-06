# Write to Row
### write_row
###

We transform a vertical stack of vectors into contiguous rows of vector-features. The native way to accomplish this in python with Pytorch tensor operations is clunky and verbose, wasteful of space, and relatively slow.