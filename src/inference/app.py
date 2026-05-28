from inferece_pipeline import NERInferencePipeline
from Normalize import Normalize
from config import Config

config = Config.from_yaml('config.yaml')

pipeline = NERInferencePipeline(
    model_path=config.output_dir,
    labels_path=config.output_file
)

pipeline.load()

text = '''Меня зовут Елена, мне 34 года, и я профессиональный повар с одиннадцатилетним стажем. Начинала помощником в небольшом кафе, за два года выросла до шеф-повара. С тех пор работала в четырёх ресторанах разного уровня — от демократичных итальянских до авторской кухни с мишленовским шефом. Особенно люблю выпечку и десерты, но уверенно чувствую себя на горячем участке и на холодном цехе. Постоянно учусь: прошла курсы по молекулярной кухне, веганской гастрономии и работе с темперированием шоколада. Важно для меня — чистота на кухне, организация процессов и умение быстро принимать решения в час пик. Легко нахожу язык с командой, могу обучить новичков. Ищу место с достойной зарплатой и интересным меню, где ценят творческий подход'''

result = pipeline.predict(text)

norm = Normalize(result)#post-processing text

norm.normalize()