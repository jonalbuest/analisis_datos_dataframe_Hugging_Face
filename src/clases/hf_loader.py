
class HfLoader:
    
    """Clase para cargar y procesar datos desde Hugging Face.
    
    Args:
        dataset_name (str): Nombre del dataset en Hugging Face.
        split (str): División del dataset a cargar (por ejemplo, 'train', 'test').
        columns (list): Lista de columnas a seleccionar del dataset.
        limit (int): Número máximo de registros a cargar.
        
        """