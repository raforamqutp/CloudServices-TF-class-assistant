#!/bin/bash
set -e
PG_CONF=$(find /etc/postgresql -name postgresql.conf | head -n 1)
sed -i 's/^port = 5432/port = 5433/' "$PG_CONF"
service postgresql restart
ss -tlnp | grep 5433 || true
echo "PostgreSQL escuchando en 5433!"
