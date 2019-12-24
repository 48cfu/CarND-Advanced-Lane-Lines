### author @48cfu
## Docker and GUI
- To activate GUI with X server Windows: https://blogs.msdn.microsoft.com/jamiedalton/2018/05/17/windows-10-docker-gui/ i.e.  Xming X Server, an X11 display server for Microsoft Windows

## If system is not yet configured
Follow this instruction, otherwise skip to next step
- docker images
- docker system prune
- docker build -t 48cfu_nanodegree .

## After setting up the system run like
Navidate to project folder. With data/, installation/ example/ etc..
- docker run --privileged -it -v ${pwd}:/src -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=192.168.0.5:0.0 48cfu_nanodegree

## For jupiter notebook
- docker run --privileged -it -v ${pwd}:/src -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY -p 8888:8888 48cfu_nanodegree
- cd ..
- ./run.sh


## TENSOR FLOW
To activate this environment, use:
- source activate carnd-term1

To deactivate an active environment, use:
- source deactivate


## HELP
- https://stackoverflow.com/questions/35595766/matplotlib-line-magic-causes-syntaxerror-in-python-script
