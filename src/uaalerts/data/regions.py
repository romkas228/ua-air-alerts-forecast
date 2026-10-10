"""Canonical region (oblast-level) identifiers and name mappings for all sources."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    id: str
    name_en: str  # as in the Vadimkin dataset
    name_uk: str  # as in alerts.in.ua


REGIONS: tuple[Region, ...] = (
    Region("cherkaska", "Cherkaska oblast", "Черкаська область"),
    Region("chernihivska", "Chernihivska oblast", "Чернігівська область"),
    Region("chernivetska", "Chernivetska oblast", "Чернівецька область"),
    Region("crimea", "Autonomous Republic of Crimea", "Автономна Республіка Крим"),
    Region("dnipropetrovska", "Dnipropetrovska oblast", "Дніпропетровська область"),
    Region("donetska", "Donetska oblast", "Донецька область"),
    Region("ivano_frankivska", "Ivano-Frankivska oblast", "Івано-Франківська область"),
    Region("kharkivska", "Kharkivska oblast", "Харківська область"),
    Region("khersonska", "Khersonska oblast", "Херсонська область"),
    Region("khmelnytska", "Khmelnytska oblast", "Хмельницька область"),
    Region("kirovohradska", "Kirovohradska oblast", "Кіровоградська область"),
    Region("kyiv_city", "Kyiv City", "м. Київ"),
    Region("kyivska", "Kyivska oblast", "Київська область"),
    Region("luhanska", "Luhanska oblast", "Луганська область"),
    Region("lvivska", "Lvivska oblast", "Львівська область"),
    Region("mykolaivska", "Mykolaivska oblast", "Миколаївська область"),
    Region("odeska", "Odeska oblast", "Одеська область"),
    Region("poltavska", "Poltavska oblast", "Полтавська область"),
    Region("rivnenska", "Rivnenska oblast", "Рівненська область"),
    Region("sumska", "Sumska oblast", "Сумська область"),
    Region("ternopilska", "Ternopilska oblast", "Тернопільська область"),
    Region("vinnytska", "Vinnytska oblast", "Вінницька область"),
    Region("volynska", "Volynska oblast", "Волинська область"),
    Region("zakarpatska", "Zakarpatska oblast", "Закарпатська область"),
    Region("zaporizka", "Zaporizka oblast", "Запорізька область"),
    Region("zhytomyrska", "Zhytomyrska oblast", "Житомирська область"),
)

REGION_IDS: tuple[str, ...] = tuple(r.id for r in REGIONS)
BY_NAME_EN: dict[str, str] = {r.name_en: r.id for r in REGIONS}
BY_NAME_UK: dict[str, str] = {r.name_uk: r.id for r in REGIONS}
