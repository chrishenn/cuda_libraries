# Collapsing Neighbors
### coll_nebs
###

Given a 4D batch of images [B,C,h,w] with representative objects pts [y,x,z,size ...] and img indexes imgid [imgindex B], we collapse nearby objects into stable neighborhood points. We iteratively apply mutual attractive force between neighboring objects, with some dampening factor and spring factor. Given that we're prototyping this behavior, there are no tests; dampening and spring constants would require tuning for given densities and sizes of objects.

In this implementation we use an exponentially-scaled measure, dependent on objects' sizes, to determine neighbor status. 