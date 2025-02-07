# Use a lightweight base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy everything from the current directory to /app in the container
COPY . /app

# Ensure the script is executable
RUN ./install_system_packages.sh
RUN ./install_git_modules.sh
RUN ./install_pip3_packages.sh
RUN ./install_cpp.sh
RUN ./install_thirdparty.sh


RUN mkdir output

# Run the script
ENTRYPOINT ["python","main_slam.py"]

