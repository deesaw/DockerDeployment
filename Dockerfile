FROM continuumio/anaconda3:4.4.0
MAINTAINER UNP, https://unp.education
EXPOSE 8000
RUN apt-get update && apt-get install -y apache2 \
    apache2-dev \   
    vim \
 && apt-get clean \
 && apt-get autoremove \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /var/www/tmt_main/
COPY ./tmt_main.wsgi /var/www/tmt_main/tmt_main.wsgi
COPY ./flask_demo /var/www/tmt_main/
RUN pip install -r requirements.txt
RUN /opt/conda/bin/mod_wsgi-express install-module
RUN mod_wsgi-express setup-server tmt_main.wsgi --port=8000 \
    --user www-data --group www-data \
    --server-root=/etc/mod_wsgi-express-80
CMD /etc/mod_wsgi-express-80/apachectl start -D FOREGROUND
