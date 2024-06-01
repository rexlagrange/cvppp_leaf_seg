# cvppp_leaf_seg
source code for cvppp leaf segmentation

Ultralytics support Ghostconv in model yaml file by default, while BiFPN need some injection.
To enable the support of BiFPN follow these steps:

1 install ultralytics

2 copy peng_injection_bifpn to ultralytics folder 

3 edit tasks.py in nn foler

  add "from ultralytics.peng_injection_bifpn.parse_model import parse_model"
  
  then rename or delete the original parse_model function in task.py
