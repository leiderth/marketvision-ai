# ExecutionManager — estado en este repositorio

Este módulo define **solo la interfaz abstracta** (`ExecutionManagerBase`) que
cualquier motor de ejecución tendría que implementar. **No incluye ninguna
implementación que ejecute operaciones reales contra Quotex ni contra ningún
otro bróker de opciones binarias.**

## Por qué

Los instrumentos de opciones binarias / expiración corta están prohibidos o
restringidos para clientes minoristas en varias jurisdicciones. Automatizar
la ejecución de este tipo de instrumento sin confirmación humana por
operación es exactamente el escenario que más riesgo legal y financiero
concentra de todo el proyecto (ver aviso en el README raíz y sección 0/13 de
la especificación técnica original).

## Qué sí está implementado

- `ExecutionManagerBase`: interfaz con los métodos que la especificación
  define (`submit_order`, `cancel_order`, `get_open_positions`, `kill_switch`),
  todos `NotImplementedError` en esta clase base.
- `PaperExecutionManager` (ver `backend/paper_trading/`, Fase 9): implementación
  que simula órdenes sin dinero real — este es el motor que sí queda operativo.

## Qué NO está implementado, ni se implementará por defecto

- Cualquier `LiveExecutionManager` que envíe órdenes reales a un bróker.

Si en algún momento decides construir esa pieza:

1. Confirma la legalidad del producto y del modo de operación en tu
   jurisdicción y la de tus usuarios.
2. El `RiskManager` (backend/risk/) debe seguir siendo la única fuente de
   verdad sobre si se puede operar — ningún `LiveExecutionManager` puede
   sortearlo.
3. Implementa primero el kill switch y el escalado gradual de capital
   (sección 13.2 de la spec) antes de conectar cualquier credencial real.
