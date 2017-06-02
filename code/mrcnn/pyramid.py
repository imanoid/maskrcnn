import tensorflow as tf
import abc

class SegmentFactory(object):
    @abc.abstractmethod
    def build_segment(self,
                      input, # input to segment
                      skip_connection_in=None # incoming skip connection
                      ):
        pass

def create_upsampling_pyramid(segment_tails, # list of segment outputs, ordered
                              segment_factory # used for creating segments between the output featuremaps
                              ):
    output_featuremaps = []
    output_featuremaps.append(segment_tails[-1])
    output_featuremap = segment_tails[-1]

    for segment_tail in reversed(segment_tails[:-2]):
        output_featuremap = segment_factory.build_segment(output_featuremap, segment_tail)
