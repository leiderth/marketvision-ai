# MLModelManager + LearningManager (Fases 7 y 11)

## Fase 7: modelo supervisado y registro local

La primera implementación usa regresión logística binaria de scikit-learn con
cinco características versionadas (`FEATURE_NAMES`) derivadas de indicadores
ya validados. Entrena solo mediante comando local explícito:

```powershell
cd backend
python -m ml.train C:\ruta\al\dataset.csv --registry .\models\registry
```

El CSV requiere `timestamp`, `label_end_timestamp`, las cinco características
del contrato y `label` (`CALL` o `PUT`). Los datos se ordenan cronológicamente,
se separan 60/20/20 en train/validation/test y se purgan los ejemplos cuya
ventana de resultado cruza al siguiente bloque. Se exige un mínimo de 100
observaciones originales, ambas etiquetas por partición y balanced accuracy
de al menos 0.55 en validation y test. Un resultado que no supera los umbrales
no se registra. Las etiquetas deben corresponder a resultados históricos
observados, no a señales generadas por el propio modelo; documenta el horizonte
de etiqueta y evita datos con look-ahead.

El registro escribe artefactos JSON atómicos con coeficientes, escalado,
esquema, métricas, versión e integridad SHA-256. No carga pickle/joblib ni
entrena durante el arranque. Un modelo nuevo queda **inactivo**; después de
revisar sus datos y métricas, el operador puede seleccionar explícitamente su
versión exacta con `ML_ACTIVE_MODEL_VERSION`. No hay promoción automática.
`ML_MODEL_REGISTRY_DIR` selecciona el directorio y
`ML_MIN_CONFIDENCE` define el umbral de puntuación de clase (por defecto 0.65).
La puntuación logística no está calibrada y no es una probabilidad fiable de
éxito.

Si se configura un modelo activo, debe cargar y pasar verificación; ausencia,
integridad inválida, características faltantes, puntuación baja o desacuerdo
con el SignalEngine determinista fuerzan `NO_TRADE`. El modelo solo puede
confirmar, nunca reemplazar los vetos de feed/calidad/régimen ni autorizar
ejecución. `RiskManager` continúa como veto independiente e inmutable.

Las explicaciones ML usan la atribución Shapley lineal interventional exacta
en espacio log-odds para el modelo lineal y la distribución de referencia
guardada en el entrenamiento. Esta explicación depende del supuesto de
independencia interventional; no implica causalidad ni debe confundirse con
una confianza probabilística calibrada.

## Fase 11: monitor de drift

Al cargar una versión explícita, `FeatureDriftMonitor` observa características
solo en timestamps distintos y compara el promedio móvil de hasta 60 muestras
contra la media/desviación del entrenamiento. Con 30 muestras, una desviación
de al menos 2.5 escalas en cualquier característica marca drift y lo mantiene
bloqueado hasta activar otra versión explícitamente. Es un detector de cambio
de media, no cubre todo drift de distribución/concepto; no hay aprendizaje ni
reentrenamiento online, shadow promotion ni modificación de RiskManager. Las
muestras y el bloqueo quedan persistidos por versión en
`ML_DRIFT_STATE_DIR`; corrupción del estado también bloquea la inferencia.
Las primeras 30 velas distintas tras activar un modelo permanecen en
NO_TRADE mientras se llena la ventana mínima de monitorización.

La Fase 7 no activa ningún modelo ni incluye datos sintéticos. La selección de
datos y evaluación más extensa requieren registros históricos observados.
Este módulo nunca puede cambiar límites del `RiskManager`.
