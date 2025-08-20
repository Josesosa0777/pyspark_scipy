"""
from pyspark.sql import SparkSession

# Crear SparkSession
spark = SparkSession.builder \
    .appName("MiAplicacionSpark") \
    .getOrCreate()
print("SparkSession creada con éxito")
spark_context = spark.sparkContext
datos = [1, 2, 3, 4, 5, 6]
rdd = spark_context.parallelize(datos, numSlices=2)
print(rdd.collect())
num_pares = rdd.filter(lambda x: x % 2 == 0)
print(num_pares.collect())
spark.stop()


from pyspark.sql import SparkSession
spark = SparkSession.builder \
            .appName ("AppSpark") \
            .getOrCreate()
spark_context = spark.sparkContext
ruta_archivo = "datos.csv"
rdd = spark_context.textFile(ruta_archivo)
if rdd.isEmpty():
    print("El RDD está vacío. Verifica el contenido del archivo.") 
else:
    fila_encabezado = rdd.first()
    datos_sin_encabezado = rdd.filter(lambda fila: fila != fila_encabezado)
    datos_sin_encabezado = datos_sin_encabezado.map(lambda fila: fila.split(","))
    print("Primeros 5 registros:")
    # ⚠️ Cuidado con .collect(). Si estás trabajando con archivos muy grandes (miles o millones de filas), 
    # puede consumir toda tu RAM y hacer que tu programa se bloquee.
    for registro in datos_sin_encabezado.take(5): # .collect() trae todos los datos del RDD a la memoria local (tu computadora).
        print(registro)
spark.stop()

from pyspark.sql import SparkSession
spark = SparkSession.builder \
            .appName ("AppSpark") \
            .getOrCreate()
spark_context = spark.sparkContext
envios_rdd = spark_context.parallelize([
    ("ZX123", "Envío a Cartago"),
    ("AB456", "Envío a Heredia"),
    ("MN789", "Envío a Alajuela"),
    ("CD234", "Envío a San José"),
    ("XY567", "Envío a Puntarenas")
])
envios_ordenados_rdd = envios_rdd.sortByKey()
print("Envíos ordenados por código de seguimiento de manera ascendente:")
for codigo, detalle in envios_ordenados_rdd.collect():
    print(f"{codigo}: {detalle}")
spark.stop()


from pyspark.sql import SparkSession
spark = SparkSession.builder \
            .appName ("AppSpark") \
            .getOrCreate()
spark_context = spark.sparkContext
clientes_rdd = spark_context.parallelize([
    ("C001", "Ana"),
    ("C002", "Carlos"),
    ("C003", "Lucía"),
    ("C004", "Pedro")
])
contratos_rdd = spark_context.parallelize([
    ("C001", "Contrato Luz"),
    ("C002", "Contrato Televisión"),
    ("C003", "Contrato Internet"),
    ("C005", "Contrato Telefonía")
])
clientes_contratos_rdd = clientes_rdd. join (contratos_rdd)
print("Clientes y sus contratos combinados:")
for cliente_id, (nombre, contrato) in clientes_contratos_rdd.collect():
    print(f"ID: {cliente_id}, Nombre: {nombre}, Contrato: {contrato}")
spark.stop()


from pyspark.sql import SparkSession
spark = SparkSession.builder \
            .appName ("AppSpark") \
            .getOrCreate()
spark_context = spark.sparkContext
observaciones_rdd = spark_context. parallelize([
    ("Eclipse", 90),
    ("Cometa", 70),
    ("Eclipse", 85),
    ("Supernova", 95),
    ("Cometa", 75)
])
print("Primer registro de observación:", observaciones_rdd.first())
print("Total de observaciones: ", observaciones_rdd.count())
print("Conteo por fenómeno: ", observaciones_rdd.countByKey())
spark.stop()


from pyspark.sql import SparkSession
spark = SparkSession.builder \
            .appName ("AppSpark") \
            .getOrCreate()
spark_context = spark.sparkContext
transacciones_rdd = spark_context.parallelize([100, 200, 300, 400, 500])
total_transacciones = transacciones_rdd. reduce (lambda x, y: x - y)
print("Total de transacciones del día:", total_transacciones)
spark.stop()

"""

from pyspark.sql import SparkSession
spark = SparkSession.builder \
            .appName ("AppSpark") \
            .getOrCreate()
spark_context = spark.sparkContext
datos = [("Juan", 28, "Ingeniero"),
    ("Ana", 23, "Diseñadora"),
    ("Luis", 35, "Arquitecto"),
    ("Carlos", 45, "Analista"),
    ("Laura", 30, "Desarrolladora"),
    ("Mateo", 25, "Diseñador"),
    ("Sofía", 38, "Gerente de Proyecto"),
    ("Pedro", 50, "Director"),
    ("Elena", 27, "Marketing"),
    ("José" ,33, "Consultor"),
    ("Lucía", 29, "Administradora")]
nombre_columnas = ["Nombre", "Edad", "Profesión"]
df = spark.createDataFrame(datos, nombre_columnas)

# df = spark.read.csv("datos.csv", header=True, inferSchema=True)
df.show()
# Filtrar filas donde la edad es mayor a 30 
df_filtrado = df.filter(df["Edad"] > 30)
df_filtrado.show()
# Same result as above but using where
df_filtrado_where = df.where(df["Edad"] > 30)
df_filtrado_where.show()

df_filtrado_where.select("Nombre", "Profesión").show()
spark.stop()



from pyspark.sql import SparkSession
from pyspark.sql. functions import lower, upper, concat_ws

spark = SparkSession.builder \
            .appName ("AppSpark") \
            .getOrCreate()
datos = [("Gen A", "Muestra", 10),
        ("gen b", "Muestra", 20),
        ("gen C", "Muestra", 30)]
nombre_columnas = ["Nombre_Gen", "Tipo", "Valor"]
df = spark. createDataFrame(datos, nombre_columnas)
df_transformado = df.withColumn("Nombre_Gen", upper(df["Nombre_Gen"]))
df_transformado = df_transformado.withColumn("Muestra_ID", concat_ws("_", lower(df["Tipo"]), df["Valor"]))
df_transformado.show()
spark.stop()



from pyspark.sql import SparkSession
from pyspark.sql. functions import sum, avg, max, min

spark = SparkSession.builder \
            .appName ("AppSpark") \
            .getOrCreate()

datos = [("Enero", 200, "Jose"),
        ("Enero", 450, "Maria"),
        ("Febrero", 300, "Jose"),
        ("Febrero", 700, "Maria"),
        ("Marzo", 500, "Juan"),]

nombre_columnas = ["Mes", "Ventas", "Vendedor"]
df = spark.createDataFrame(datos, nombre_columnas)
df_agg = df.groupBy("Mes").agg(
    sum("Ventas").alias("TotalVentas"),
    avg("Ventas").alias("PromedioVentas"),
    max("Ventas").alias("MaxVenta"),
    min("Ventas").alias("MinVenta")
)
df_agg.show()
spark.stop()



from pyspark.sql import SparkSession

spark = SparkSession.builder \
            .appName ("AppSpark") \
            .getOrCreate()

datos = [("Juan", 28, "Ingeniero"),
    ("Ana", 23, "Diseñadora"),
    ("Luis", 35, "Arquitecto"),
    ("Carlos", 45, "Analista"),
    ("Laura", 30, "Desarrolladora"),
    ("Mateo", 25, "Diseñador"),
    ("Sofía", 38, "Gerente de Proyecto"),
    ("Pedro", 50, "Director"),
    ("Elena", 27, "Marketing"),
    ("José", 33, "Consultor"),
    ("Lucía", 29, "Administradora")]
columnas = ["Nombre", "Edad", "Profesión"]
df = spark.createDataFrame(datos, columnas)
df_ordenado = df.sort("Nombre")
df_ordenado.show()
df_ordenado_desc = df.orderBy(df["Edad"].desc())
df_ordenado_desc.show()
spark.stop()



from pyspark.sql.functions import when
spark = SparkSession.builder \
            .appName ("AppSpark") \
            .getOrCreate()

datos = [("Juan", 28),
    ("Ana", 23),
    ("Luis", 35),
    ("Carlos" ,45),
    ("Laura", 30),
    ("Mateo", 25),
    ("Sofía", 38),
    ("Pedro" ,50),
    ("Elena", 27),
    ("José", 33),
    ("Lucía", 29)]
nombre_columnas = ["Nombre", "Horas"]
df = spark. createDataFrame(datos, nombre_columnas)
df_condicional = df.withColumn("Horas Extra", when(df["Horas"] > 40, df["Horas"] - 40).otherwise(0))
df_condicional.show()
spark.stop()


from pyspark.sql import SparkSession
spark = SparkSession.builder \
        .appName("AppSpark") \
        .getOrCreate()

clientes = [("1", "Juan", "España"),
            ("2", "Ana", "México"), 
            ("3", "Luis", "Colombia")]
compras = [("1", "Teléfono", 1000),
            ("2", "Computadora", 1500),
            ("1", "Monitor", 550),
            ("4", "Audífonos", 360)]

df_clientes = spark.createDataFrame(clientes, ["Cliente_ID", "Nombre", "País"])
df_compras = spark.createDataFrame(compras, ["Cliente_ID", "Producto", "Monto"])
df_join = df_clientes.join(df_compras, on="Cliente_ID", how="inner")
df_join.show()
spark.stop()