#!/bin/sh
set -euo pipefail

CERT_FULLCHAIN="/etc/letsencrypt/live/${SSL_DOMAIN}/fullchain.pem"
CERT_PRIVKEY="/etc/letsencrypt/live/${SSL_DOMAIN}/privkey.pem"
TEMPLATE_DIR="/etc/nginx/templates"
CONF_OUTPUT="/etc/nginx/conf.d/default.conf"

if [ -f "$CERT_FULLCHAIN" ] && [ -f "$CERT_PRIVKEY" ]; then
  echo "[gateway] Using HTTPS-enabled nginx config."
  envsubst '${SSL_DOMAIN} ${SSL_DOMAIN_ALIASES} ${SSL_CERTBOT_WEBROOT}' \
    < "$TEMPLATE_DIR/default.conf.template" \
    > "$CONF_OUTPUT"
else
  echo "[gateway] Certificate not found. Using HTTP bootstrap config."
  envsubst '${SSL_DOMAIN} ${SSL_DOMAIN_ALIASES} ${SSL_CERTBOT_WEBROOT}' \
    < "$TEMPLATE_DIR/default-http.conf.template" \
    > "$CONF_OUTPUT"
fi

exec nginx -g 'daemon off;'
