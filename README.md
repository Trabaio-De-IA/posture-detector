# Posture Detector — Detector de Postura com Inteligência Artificial

O **Posture Detector** é uma aplicação de detecção de postura em tempo real utilizando Inteligência Artificial e Visão Computacional.

O sistema utiliza a webcam do usuário para capturar os movimentos corporais, identificar pontos do corpo humano e analisar se a postura está adequada ou inadequada. A aplicação foi desenvolvida com foco em postura sentada, especialmente para usuários que passam muito tempo em frente ao computador.

O projeto combina Python, Flask, MediaPipe, OpenCV, NumPy, HTML, CSS e JavaScript.

 

## Objetivo do projeto

O objetivo deste projeto é detectar automaticamente problemas posturais durante o uso do computador, como:

* pescoço inclinado;
* tronco curvado;
* cabeça projetada para frente;
* cabeça baixa;
* ombros desalinhados para a direita ou esquerda;
* postura ruim mantida por vários frames consecutivos.

Quando o sistema identifica uma postura inadequada por um determinado período, ele classifica a postura como **Ruim** e pode emitir alertas visuais e sonoros para o usuário.

 

## Tecnologias utilizadas

O projeto utiliza as seguintes tecnologias:

* Python 3.12
* Flask
* MediaPipe
* OpenCV
* NumPy
* HTML
* CSS
* JavaScript
* uv
* Webcam do navegador
* SpeechSynthesis API para alertas de voz

 

## Como o sistema funciona

O funcionamento geral do sistema ocorre em etapas:

1. O usuário acessa a aplicação pelo navegador.
2. O navegador solicita permissão para acessar a webcam.
3. A câmera captura os frames em tempo real.
4. O modelo de IA identifica pontos corporais do usuário.
5. O sistema calcula métricas posturais com base nos pontos detectados.
6. Os valores calculados são comparados com thresholds definidos no código.
7. A postura é classificada como **Boa** ou **Ruim**.
8. Caso a postura ruim permaneça por vários frames, o sistema exibe e/ou fala um alerta.

 
## Métricas analisadas

O sistema pode analisar diferentes aspectos da postura:

| Métrica                      | Descrição                                                                         |
|          - |                             |
| Inclinação do pescoço        | Verifica se a cabeça/pescoço está inclinado em relação ao corpo                   |
| Inclinação do tronco         | Verifica se o tronco está inclinado                                               |
| Cabeça projetada para frente | Usa profundidade para identificar se a cabeça está avançada em relação aos ombros |
| Cabeça baixa                 | Verifica se o usuário está olhando muito para baixo                               |
| Ombros inclinados            | Verifica se um ombro está mais alto ou mais baixo que o outro                     |


## Estrutura do repositório

A estrutura principal do projeto é:

```txt
posture-detector/
├── app/
├── pyproject.toml
├── run.py
├── uv.lock
└── README.md

 

Descrição dos principais itens:

| Arquivo/Pasta    | Descrição                                               |
|      - |                   - |
| `app/`           | Pasta principal da aplicação                            |
| `run.py`         | Arquivo usado para iniciar o servidor Flask             |
| `pyproject.toml` | Arquivo de configuração do projeto e dependências       |
| `uv.lock`        | Arquivo de controle das dependências gerenciado pelo uv |
| `README.md`      | Documentação do projeto                                 |

 
```
## Pré-requisitos

Antes de executar o projeto, é necessário ter instalado:

* Python 3.12
* Git
* uv
* Navegador com suporte à webcam
* Webcam funcionando

## Instalação do Python

O projeto requer Python na versão 3.12.

Verifique a versão instalada com:

```bash
python --version
ou:
py --version
```
 
O resultado ideal deve ser semelhante a:

```txt
Python 3.12.x
````
Caso o terminal mostre Python 3.11, mas o ambiente virtual do VS Code mostre Python 3.12, execute o projeto pelo ambiente virtual correto.

## Instalação do uv

Este projeto pode ser executado com o **uv**, um gerenciador moderno para projetos Python.

A documentação oficial do uv está disponível em:

```txt
https://docs.astral.sh/uv/
````
Para instalar o uv no Windows, execute no PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
````
Depois de instalar, feche e abra novamente o terminal.

Verifique se o uv foi instalado corretamente:

```bash
uv --version
 
````
## Como executar o projeto

Este projeto utiliza **uv** para gerenciar o ambiente Python e instalar as dependências.

Com o `uv`, não é necessário criar ou ativar ambiente virtual manualmente.

### 1. Clone o repositório

````bash
git clone https://github.com/Trabaio-De-IA/posture-detector.git
````

### 2. Entre na pasta do projeto

````bash
cd posture-detector
````

### 3. Instale as dependências


````bash
pip install mediapipe opencv-python flask numpy

uv sync
````

Esse comando cria o ambiente virtual automaticamente e instala todas as dependências do projeto.

### 4. Execute a aplicação

````bash
uv run run.py
````

### 5. Acesse no navegador

````txt
http://localhost:5000
````

Ao abrir a página, permita o acesso à câmera.

## Dependências principais

As dependências principais do projeto são:

| Dependência | Função                                       |
|    -- |               -- |
| MediaPipe   | Detecção dos pontos corporais                |
| OpenCV      | Processamento de imagem e vídeo              |
| Flask       | Servidor web da aplicação                    |
| NumPy       | Operações matemáticas e manipulação de dados |

 

## Configurações de postura

O sistema utiliza thresholds para definir quando uma postura deve ser considerada ruim.

Exemplo de configuração:

```js
const BAD_NECK_THRESHOLD = 30.0;
const BAD_TORSO_THRESHOLD = 12;
const BAD_HEAD_FWD = 0.7;
const BAD_HEAD_DOWN = 0.3;
const BAD_SHOULDER_TILT_THRESHOLD = 8.0;
const CONSECUTIVE_BAD_FRAMES = 15;

 
````
Descrição:

| Constante                     | Descrição                                                       |
|          -- |                       |
| `BAD_NECK_THRESHOLD`          | Limite para inclinação do pescoço                               |
| `BAD_TORSO_THRESHOLD`         | Limite para inclinação do tronco                                |
| `BAD_HEAD_FWD`                | Limite para cabeça projetada para frente                        |
| `BAD_HEAD_DOWN`               | Limite para cabeça baixa                                        |
| `BAD_SHOULDER_TILT_THRESHOLD` | Limite para ombros desalinhados                                 |
| `CONSECUTIVE_BAD_FRAMES`      | Quantidade de frames ruins seguidos para confirmar postura ruim |

 

## Detecção de ombros inclinados

Uma das melhorias implementadas no projeto é a detecção da inclinação dos ombros.

A lógica utiliza os landmarks dos ombros:

```js
lm[11] // ombro esquerdo
lm[12] // ombro direito

 
````
O sistema calcula o ângulo da linha entre os dois ombros em relação à horizontal.

Se o ângulo ultrapassar o limite definido, o sistema identifica que os ombros estão desalinhados:

```js
const BAD_SHOULDER_TILT_THRESHOLD = 8.0;

 
````
A função utilizada para calcular esse ângulo é:

```js
function horizontalAngle(leftPoint, rightPoint) {
return (
Math.atan2(
Math.abs(leftPoint.y - rightPoint.y),
Math.abs(leftPoint.x - rightPoint.x)
) *
(180 / Math.PI)
);
}

 
````
A verificação pode ser feita assim:

```js
const shoulderTiltAngle = horizontalAngle(lm[11], lm[12]);

const isShoulderTiltBad =
shoulderTiltAngle > BAD_SHOULDER_TILT_THRESHOLD;

 
````
E incluída na regra geral de postura ruim:

```js
const isBad =
neckAngle > BAD_NECK_THRESHOLD ||
torsoAngle > BAD_TORSO_THRESHOLD ||
headFwd > BAD_HEAD_FWD ||
headDown > BAD_HEAD_DOWN ||
isShoulderTiltBad;

 

 
````
## Classificação da postura

A postura é classificada como ruim quando qualquer uma das métricas ultrapassa o limite definido.

Exemplo:

```js
const isBad =
neckAngle > BAD_NECK_THRESHOLD ||
torsoAngle > BAD_TORSO_THRESHOLD ||
headFwd > BAD_HEAD_FWD ||
headDown > BAD_HEAD_DOWN ||
isShoulderTiltBad;

badCount = isBad ? badCount + 1 : 0;

const status = badCount >= CONSECUTIVE_BAD_FRAMES ? "Ruim" : "Boa";

 
````
Esse controle evita que movimentos rápidos ou pequenas variações sejam classificados imediatamente como postura ruim.

 

## Alertas de voz

O sistema possui alertas de voz para avisar o usuário quando a postura estiver inadequada.

Exemplo de mensagens:

```js
const ALERT_MESSAGES = [
"Por favor, corrija sua postura! Você está curvado.",
"Atenção! Sua postura está incorreta. Sente-se ereto.",
"Lembrete de postura: endireite as costas e levante a cabeça.",
"Cuide da sua saúde! Corrija sua postura agora.",
];

 

O intervalo entre alertas é controlado por:

const ALERT_COOLDOWN_MS = 30_000;

 
````
Nesse exemplo, o sistema aguarda 30 segundos antes de emitir outro alerta.

 

## K-Means

O projeto também pode coletar dados da sessão para agrupar padrões posturais usando K-Means.

A coleta pode armazenar métricas como:

```js
sessionData.push([neckAngle, torsoAngle]);

````

Com isso, o sistema pode classificar padrões como:

* postura ereta;
* pescoço inclinado;
* tronco curvado;
* postura intermediária.

 

## Recomendações para melhor uso

Para melhorar a precisão da detecção:

* posicione a câmera na altura dos olhos;
* sente-se de frente para a câmera;
* mantenha cabeça, ombros e tronco visíveis;
* evite ambientes escuros;
* mantenha uma distância adequada da câmera;
* evite que a câmera fique muito inclinada;
* mantenha o notebook ou webcam em uma posição estável.

 

## Problemas comuns

### O comando `python --version` mostra Python 3.11

O projeto requer Python 3.12.

No Windows, tente:

```bash
py -3.12 --version

````
Se estiver usando uv, execute:

```bash
uv run python --version 
````
 

### O comando `uv` não é reconhecido

Feche e abra novamente o terminal.

Depois teste:

```bash
uv --version

````
Se ainda não funcionar, instale novamente o uv seguindo a documentação:

```txt
https://docs.astral.sh/uv/
 
````
 

### A câmera não abre

Verifique se:

* o navegador tem permissão para acessar a câmera;
* outro aplicativo não está usando a câmera;
* a câmera está funcionando corretamente;
* o servidor Flask está rodando;
* você acessou `http://localhost:5000`.

 

### Erro: módulo não encontrado

Se aparecer erro informando que algum módulo não foi encontrado, execute novamente:

```bash
uv sync

````
Se estiver usando pip manualmente:

```bash
pip install mediapipe opencv-python flask numpy

````
 

### A postura está sensível demais

Aumente um pouco os thresholds.

Exemplo:

```js
const BAD_HEAD_FWD = 0.8;
const BAD_HEAD_DOWN = 0.35;
const BAD_SHOULDER_TILT_THRESHOLD = 10.0;
const CONSECUTIVE_BAD_FRAMES = 18;

````
 

### A postura está pouco sensível

Reduza um pouco os thresholds.

Exemplo:

```js
const BAD_HEAD_FWD = 0.6;
const BAD_HEAD_DOWN = 0.25;
const BAD_SHOULDER_TILT_THRESHOLD = 6.0;
const CONSECUTIVE_BAD_FRAMES = 10;

````
 

## Comandos principais

Instalar dependências:

```bash
uv sync

````
Executar o projeto:

```bash
uv run run.py

````
Acessar aplicação:

```txt
http://localhost:5000

````
## Possíveis melhorias futuras

Algumas melhorias que podem ser implementadas futuramente:

* tela de configuração dos thresholds;
* calibração inicial da postura ideal do usuário;
* histórico de postura durante a sessão;
* gráficos de tempo em postura boa e ruim;
* relatórios de desempenho postural;
* adaptação do K-Means para incluir inclinação dos ombros;
* alertas personalizados para cada tipo de postura ruim;
* autenticação de usuários;
* armazenamento de sessões;
* melhoria da interface visual.

 

## Status do projeto

O projeto está funcional para detecção de postura em tempo real, utilizando webcam e modelo de IA para análise corporal.

Atualmente, o sistema considera:

* inclinação do pescoço;
* inclinação do tronco;
* cabeça projetada para frente;
* cabeça baixa;
* ombros inclinados para direita ou esquerda;
* persistência da postura ruim por frames consecutivos.

 

## Autores

Projeto desenvolvido para fins acadêmicos.

Integrantes:

* Guilherme Jacinto
* Ianny Correia
* João Pedro Araujo
* Kevin Reis

 

## Licença

Este projeto foi desenvolvido para fins educacionais.
