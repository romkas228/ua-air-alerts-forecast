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


# Locations that cover each oblast's administrative centre, as named in either source
# (raion and hromada/city level). An oblast-level alert always covers the centre.
# Notes: Donetska -> Kramatorsk (de facto centre since 2014); Kyivska -> Bila Tserkva
# (largest city; the formal centre Kyiv is its own region). Crimea/Luhanska: occupied,
# oblast-level only (excluded from evaluation anyway). Kyiv City: every alert counts.
CENTER_LOCATIONS: dict[str, frozenset[str]] = {
    "cherkaska": frozenset(
        {
            "Cherkaskyi raion",
            "Черкаський район",
            "m. Cherkasy ta Cherkaska terytorialna hromada",
            "Черкаська територіальна громада",
            "м. Черкаси",
        }
    ),
    "chernihivska": frozenset(
        {
            "Chernihivskyi raion",
            "Чернігівський район",
            "m. Chernihiv ta Chernihivska terytorialna hromada",
            "Чернігівська територіальна громада",
            "м. Чернігів",
        }
    ),
    "chernivetska": frozenset(
        {
            "Chernivetskyi raion",
            "Чернівецький район",
            "m. Chernivtsi ta Chernivetska terytorialna hromada",
            "Чернівецька територіальна громада",
            "м. Чернівці",
        }
    ),
    "dnipropetrovska": frozenset(
        {
            "Dniprovskyi raion",
            "Дніпровський район",
            "m. Dnipro ta Dniprovska terytorialna hromada",
            "Дніпровська територіальна громада",
            "м. Дніпро",
        }
    ),
    "donetska": frozenset(
        {
            "Kramatorskyi raion",
            "Краматорський район",
            "m. Kramatorsk ta Kramatorska terytorialna hromada",
            "Краматорська територіальна громада",
            "м. Краматорськ",
        }
    ),
    "ivano_frankivska": frozenset(
        {
            "Ivano-Frankivskyi raion",
            "Івано-Франківський район",
            "m. Ivano-Frankivsk ta Ivano-Frankivska terytorialna hromada",
            "Івано-Франківська територіальна громада",
            "м. Івано-Франківськ",
        }
    ),
    "kharkivska": frozenset(
        {
            "Kharkivskyi raion",
            "Харківський район",
            "m. Kharkiv ta Kharkivska terytorialna hromada",
            "Харківська територіальна громада",
            "м. Харків",
        }
    ),
    "khersonska": frozenset(
        {
            "Khersonskyi raion",
            "Херсонський район",
            "m. Kherson ta Khersonska terytorialna hromada",
            "Херсонська територіальна громада",
            "м. Херсон",
        }
    ),
    "khmelnytska": frozenset(
        {
            "Khmelnytskyi raion",
            "Хмельницький район",
            "m. Khmelnytskyi ta Khmelnytska terytorialna hromada",
            "Хмельницька територіальна громада",
            "м. Хмельницький",
        }
    ),
    "kirovohradska": frozenset(
        {
            "Kropyvnytskyi raion",
            "Кропивницький район",
            "m. Kropyvnytskyi ta Kropyvnytska terytorialna hromada",
            "Кропивницька територіальна громада",
            "м. Кропивницький",
        }
    ),
    "kyivska": frozenset(
        {
            "Bilotserkivskyi raion",
            "Білоцерківський район",
            "m. Bila Tserkva ta Bilotserkivska terytorialna hromada",
            "Білоцерківська територіальна громада",
            "м. Біла Церква",
        }
    ),
    "lvivska": frozenset(
        {
            "Lvivskyi raion",
            "Львівський район",
            "m. Lviv ta Lvivska terytorialna hromada",
            "Львівська територіальна громада",
            "м. Львів",
        }
    ),
    "mykolaivska": frozenset(
        {
            "Mykolaivskyi raion",
            "Миколаївський район",
            "m. Mykolaiv ta Mykolaivska terytorialna hromada",
            "Миколаївська територіальна громада",
            "м. Миколаїв",
        }
    ),
    "odeska": frozenset(
        {
            "Odeskyi raion",
            "Одеський район",
            "m. Odesa ta Odeska terytorialna hromada",
            "Одеська територіальна громада",
            "м. Одеса",
        }
    ),
    "poltavska": frozenset(
        {
            "Poltavskyi raion",
            "Полтавський район",
            "m. Poltava ta Poltavska terytorialna hromada",
            "Полтавська територіальна громада",
            "м. Полтава",
        }
    ),
    "rivnenska": frozenset(
        {
            "Rivnenskyi raion",
            "Рівненський район",
            "m. Rivne ta Rivnenska terytorialna hromada",
            "Рівненська територіальна громада",
            "м. Рівне",
        }
    ),
    "sumska": frozenset(
        {
            "Sumskyi raion",
            "Сумський район",
            "m. Sumy ta Sumska terytorialna hromada",
            "Сумська територіальна громада",
            "м. Суми",
        }
    ),
    "ternopilska": frozenset(
        {
            "Ternopilskyi raion",
            "Тернопільський район",
            "m. Ternopil ta Ternopilska terytorialna hromada",
            "Тернопільська територіальна громада",
            "м. Тернопіль",
        }
    ),
    "vinnytska": frozenset(
        {
            "Vinnytskyi raion",
            "Вінницький район",
            "m. Vinnytsia ta Vinnytska terytorialna hromada",
            "Вінницька територіальна громада",
            "м. Вінниця",
        }
    ),
    "volynska": frozenset(
        {
            "Lutskyi raion",
            "Луцький район",
            "m. Lutsk ta Lutska terytorialna hromada",
            "Луцька територіальна громада",
            "м. Луцьк",
        }
    ),
    "zakarpatska": frozenset(
        {
            "Uzhhorodskyi raion",
            "Ужгородський район",
            "m. Uzhhorod ta Uzhhorodska terytorialna hromada",
            "Ужгородська територіальна громада",
            "м. Ужгород",
        }
    ),
    "zaporizka": frozenset(
        {
            "Zaporizkyi raion",
            "Запорізький район",
            "m. Zaporizhzhia ta Zaporizka terytorialna hromada",
            "Запорізька територіальна громада",
            "м. Запоріжжя",
        }
    ),
    "zhytomyrska": frozenset(
        {
            "Zhytomyrskyi raion",
            "Житомирський район",
            "m. Zhytomyr ta Zhytomyrska terytorialna hromada",
            "Житомирська територіальна громада",
            "м. Житомир",
        }
    ),
    "crimea": frozenset(),
    "luhanska": frozenset(),
    "kyiv_city": frozenset(),
}


def covers_center(region: str, geo_level: str, location: str | None) -> bool:
    """Whether an alert at (region, geo_level, location) covers the oblast centre."""
    if geo_level == "oblast" or region == "kyiv_city":
        return True
    return location in CENTER_LOCATIONS[region]
