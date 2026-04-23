FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/nginx.conf
COPY default.conf.template /etc/nginx/templates/default.conf.template
COPY default-http.conf.template /etc/nginx/templates/default-http.conf.template
COPY gateway-entrypoint.sh /gateway-entrypoint.sh

RUN chmod +x /gateway-entrypoint.sh

EXPOSE 80 443

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=6 \
  CMD nginx -t

ENTRYPOINT ["/gateway-entrypoint.sh"]
