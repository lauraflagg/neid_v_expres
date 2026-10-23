#!/bin/sh

#   If /bin/sh isn't available on your system, update the script to use bash instead:
#   #!/usr/bin/env bash
#
#   If you don't have a NEID archive account, just press return for the Username and Password prompts


read -p "Username: " user
read -s -p "Password : " pass

wget --save-cookies neid_cookie.txt \
      --keep-session-cookies \
      --post-data "username=${user}&password=${pass}" \
      --delete-after \
      https://neid.ipac.caltech.edu/login.php

