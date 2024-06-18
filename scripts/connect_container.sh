#!/bin/bash

docker exec -it \
  -t \
  -e "color_prompt=yes" \
  -e "TERM=xterm-256color" \
  vislang-map \
  /bin/zsh
