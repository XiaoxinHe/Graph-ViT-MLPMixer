# Problem with torch multiprocessing


# 1. What is sharing strategy and why it can faster cpu processing time? 
# Once the tensor/storage is moved to shared_memory (see share_memory_()), 
# it will be possible to send it to other processes without making any copies.


"""
Strategy 1: file_descriptor (default one)

    Note that if there will be a lot of tensors shared, this strategy will keep a large number of 
    file descriptors open most of the time. If your system has low limits for the number of open file descriptors, 
    and you can’t raise them, you should use the file_system strategy.

To raise the system limits:

    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (20000, rlimit[1]))

or: in bash,  ulimit -n 64000
"""


"""
https://github.com/pytorch/pytorch/issues/13246
Strategy 2: file_system   [faster than default]
    This has a benefit of not requiring the implementation to cache the file descriptors obtained from it, 
    but at the same time is prone to shared memory leaks.
    If your system has high enough limits, and file_descriptor is a supported strategy, we do not recommend switching to this one.

Python Multiprocessing: There is no way of storing arbitrary python objects (even simple lists) in shared memory in Python without
 triggering copy-on-write behaviour due to the addition of refcounts, everytime something reads from these objects. The refcounts 
 are added memory-page by memory-page, which is why the consumption grows slowly. The processes (workers) will end up having all/most 
 of the memory copied over bit by bit, which is why we get the memory overflow problem. Best description of this behavior is here (SO).

Possible Solution:
Using Multiprocessing like now: in order for python multiprocessing to work without these refcount effects, the objects have to 
be made “compatible with” and wrapped in multiprocessing.Array before the process pool is created and workers are forked. 
This supposedly ensures, that the memory will really be shared and no copy-on-write happens. This explains how to do it for numpy arrays 
and this explains the reasoning behind it again. Don’t get confused by some false statements even by the authors of these good answers 
stating that copy-on-write makes all of this unnecessary, which is not true. One comment also points to this:
"""