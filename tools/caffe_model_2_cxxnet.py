import sys
import re
import argparse
import time
import numpy as np
CAFFE_ROOT = "../../"
CXXNET_ROOT = "../../../cxxnet/"

sys.path.append(CXXNET_ROOT + 'wrapper/')
sys.path.append(CAFFE_ROOT + 'python/')

import caffe
import cxxnet

parser = argparse.ArgumentParser(description='Convert Caffe Model to CXXNET model')
parser.add_argument('--prototxt', required=True, help="Path to Caffe Model's deploy.prototxt")
parser.add_argument('--caffe', required=True, help="Path to Caffe Model's binary file")
parser.add_argument('--cxxconf', required=True, help="Path to CXXNET's conf file")
parser.add_argument('--out', required=True, help="Path to output CXXNET model")
parser.add_argument('--caffe_mean', default="", help="Path to Caffe mean file")
parser.add_argument('--cxx_mean', default='mean.bin', help="Path to CXXNET mean file")

args = parser.parse_args()


print '-' * 60
print "Make sure layer name is exactly same in CXXNET's conf file and Caffe's prototxt"
print "Convert will start in 5 second..."
print '-' * 60
print ""

time.sleep(5)

conv_pattern = re.compile(r'\s*layer\[.*?\]\s*=\s*conv\w*\s*')
fullc_patten = re.compile(r'\s*layer\[.*?\]\s*=\s*fullc\w*\s*')

# Load CXXNET conf and weight layer name
cxx_conf = ""
cxx_weight_layer = []

fi = file(args.cxxconf)
for line in fi:
    cxx_conf += line
    if conv_pattern.match(line) != None or fullc_patten.match(line) != None:
        line = line.split('=')[1]
        line = line.split(':')
        assert(len(line) == 2)
        line = line[1]
        name = line.split('\n')[0]
        cxx_weight_layer.append(name)

assert(len(cxx_weight_layer) > 0)

# Load CaffeNet
caffe_net = caffe.Net(args.prototxt, args.caffe, caffe.TEST)

# Get Caffe Param
params = {pr: (caffe_net.params[pr][0].data, caffe_net.params[pr][1].data) for pr in cxx_weight_layer}

# Init CXXNET model

net = cxxnet.Net(cfg = cxx_conf)
net.init_model()

# Set Weight
for pr in cxx_weight_layer:
    wmat = params[pr][0]
    bias = params[pr][1]
    net.set_weight(wmat, pr, 'wmat')
    net.set_weight(bias, pr, 'bias')

# Save Model
net.save_model(args.out)


# Convert Mean
"""
if len(args.caffe_mean) > 0:
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(args.caffe_mean , 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    out = arr[0]
    cxxnet.npy2mshadow(out, args.cxx_mean)
"""
