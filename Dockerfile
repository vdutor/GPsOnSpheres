FROM tensorflow/tensorflow:2.4.0

RUN pip install tensorflow-probability==0.12.2
RUN pip install gpflow==2.1.4 \
  matplotlib \
  meshzoo \
  plotly

# Uncomment below to copy project files into container and run command
# when container starts up.
#COPY . /GPsOnSpheres
#
# If project files will change a lot it is better to use a bind-mount.
# In this case uncomment below.
#RUN mkdir /GPsOnSpheres

ENV PYTHONPATH=/GPsOnSpheres
WORKDIR /GPsOnSpheres
ENTRYPOINT ["python"]
CMD ["scripts/two_dimensional_toy.py"]

# Do (from project root directory in host):
# docker build -t gspheres .
# docker run -u $(id -u):$(id -g) -v $(pwd):/GPsOnSpheres -it gspheres
#
# Note that CMD can be overridden by simply adding a different script to
# the docker run command, e.g.
# docker run -u $(id -u):$(id -g) -v $(pwd):/GPsOnSpheres -it gspheres scripts/plot_samples.py
