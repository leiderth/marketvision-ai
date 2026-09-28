# Observabilidad (Fase 12)

`/api/observability/health` resume el estado degradado del feed, calidad,
calendario, RiskManager y modelo; `/api/observability/metrics` expone gauges
Prometheus de solo lectura para conectividad, vetos, modelo y órdenes paper.
La salud no reemplaza los vetos por señal y nunca convierte un estado degradado
en permiso para operar. Las claves/configuración sensibles no se devuelven.

Los endpoints están diseñados para la API local de desarrollo. Antes de
exponer cualquier instancia fuera de loopback se requiere autenticación,
autorización y endurecimiento de despliegue; no exponerlos directamente a
Internet.
