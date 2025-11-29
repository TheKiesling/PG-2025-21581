# Sistema de Semáforos Inteligentes para la regulación de tránsito vehicular

## Descripción
Sistema de control de semáforos inteligentes basado en **Multi-Agent Proximal Policy Optimization (MAPPO)** con integración completa de **SUMO** (Simulation of Urban MObility). Diseñado para entrenar múltiples agentes que controlan semáforos de forma coordinada, optimizando el flujo de tráfico urbano mediante aprendizaje por refuerzo profundo con soporte GPU.

## Tecnologías Utilizadas
- **SUMO**: Simulador de movilidad urbana para modelar el tráfico vehicular
- **sumo-rl**: Biblioteca para integrar SUMO con entornos de aprendizaje por refuerzo
- **Ray RLlib**: Framework de aprendizaje por refuerzo escalable, utilizado para implementar MAPPO
- **PyTorch**: Biblioteca de aprendizaje profundo para construir y entrenar redes neuronales
- **CUDA**: Plataforma de computación paralela para acelerar el entrenamiento en GPU


## Requisitos Previos

### 1. SUMO (Simulation of Urban MObility)

**Windows:**
```bash
# Descargar e instalar desde:
# https://eclipse.dev/sumo/

# Configurar variable de entorno SUMO_HOME
# Ejemplo: C:\Program Files (x86)\Eclipse\Sumo
```

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt-get install sumo sumo-tools sumo-doc

# macOS (Homebrew)
brew install sumo

# Configurar SUMO_HOME
export SUMO_HOME="/usr/share/sumo"
```

### 2. Python 3.8+

### 3. CUDA (para soporte GPU)
- NVIDIA GPU con soporte CUDA
- CUDA Toolkit 11.8 o superior
- cuDNN compatible

## Instalación

### 1. Clonar el repositorio

```bash
git clone <tu-repositorio>
cd smart-light-system
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación de GPU (opcional)

```python
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```
## Uso

### Entrenamiento

```bash
# Entrenamiento básico
python main.py train

# Especificar número de iteraciones
python main.py train --iterations 500

# Con generación de gráficas
python main.py train --iterations 100 --plot

# Configuración personalizada
python main.py train --sumo-config config/sumo_config.yaml --training-config config/training_config.yaml
```

### Evaluación

```bash
# Evaluar modelo entrenado
python main.py evaluate --checkpoint checkpoints/checkpoint_000100

# Evaluar con más episodios
python main.py evaluate --checkpoint checkpoints/checkpoint_000100 --episodes 20
```

### Visualización

```bash
# Visualizar con SUMO-GUI
python main.py visualize --checkpoint checkpoints/checkpoint_000100

# Personalizar duración
python main.py visualize --checkpoint checkpoints/checkpoint_000100 --duration 1800
```


## Configuración

### SUMO Configuration (`config/sumo_config.yaml`)

```yaml
sumo:
  net_file: "scenarios/simple_grid/grid.net.xml"
  route_file: "scenarios/simple_grid/grid.rou.xml"
  use_gui: false              # true para visualización
  num_seconds: 3600           # Duración de la simulación
  delta_time: 5               # Segundos entre decisiones
  yellow_time: 2              # Duración de luz amarilla
  min_green: 5                # Tiempo mínimo en verde
  max_green: 60               # Tiempo máximo en verde
  reward_fn: "diff-waiting-time"  # Función de recompensa
  single_agent: false         # Multi-agente
  sumo_seed: 42               # Semilla para reproducibilidad
```

### Training Configuration (`config/training_config.yaml`)

```yaml
training:
  algorithm: "MAPPO"
  num_workers: 4              # Workers paralelos
  num_gpus: 1                 # GPUs a usar
  framework: "torch"
  
  train_batch_size: 4000
  sgd_minibatch_size: 128
  num_sgd_iter: 10
  
  lr: 0.0003                  # Learning rate
  gamma: 0.99                 # Factor de descuento
  lambda: 0.95                # GAE lambda
  clip_param: 0.2             # PPO clip
  
  model:
    fcnet_hiddens: [256, 256] # Capas ocultas
    fcnet_activation: "relu"
```

## Demo

El video demostrativo se encuentra en [/demo/demo.mp4](demo/demo.mp4)

## Documentación

El informe final del proyecto está disponible en [/docs/informe_final.pdf](docs/informe_final.pdf)

## Autor

José Pablo Kiesling Lange - 21581

## Licencia

Este proyecto está bajo la Licencia MIT.


