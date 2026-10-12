"""Land-border adjacency between the 24 panel regions.

Borders with excluded/absent regions (Luhansk, Crimea) and with foreign countries are ignored.
Kyiv City is enclaved in Kyiv oblast, so its only neighbour is Kyivska oblast.
Edges marked "unsure" were checked against general knowledge of Ukraine's map, not a GIS file.
"""

ADJACENCY: dict[str, list[str]] = {
    "Cherkaska oblast": ["Kirovohradska oblast", "Kyivska oblast", "Poltavska oblast", "Vinnytska oblast"],
    "Chernihivska oblast": ["Kyivska oblast", "Poltavska oblast", "Sumska oblast"],
    "Chernivetska oblast": ["Ivano-Frankivska oblast", "Khmelnytska oblast", "Ternopilska oblast",
                            "Vinnytska oblast"],  # unsure: short Chernivtsi-Vinnytsia border near Sokyriany
    "Dnipropetrovska oblast": ["Donetska oblast", "Kharkivska oblast", "Khersonska oblast", "Kirovohradska oblast",
                               "Mykolaivska oblast", "Poltavska oblast", "Zaporizka oblast"],
    "Donetska oblast": ["Dnipropetrovska oblast", "Kharkivska oblast", "Zaporizka oblast"],
    "Ivano-Frankivska oblast": ["Chernivetska oblast", "Lvivska oblast", "Ternopilska oblast", "Zakarpatska oblast"],
    "Kharkivska oblast": ["Dnipropetrovska oblast", "Donetska oblast", "Poltavska oblast", "Sumska oblast"],
    "Khersonska oblast": ["Dnipropetrovska oblast", "Mykolaivska oblast", "Zaporizka oblast"],
    "Khmelnytska oblast": ["Chernivetska oblast", "Rivnenska oblast", "Ternopilska oblast", "Vinnytska oblast",
                           "Zhytomyrska oblast"],
    "Kirovohradska oblast": ["Cherkaska oblast", "Dnipropetrovska oblast", "Mykolaivska oblast", "Odeska oblast",
                             "Poltavska oblast", "Vinnytska oblast"],  # unsure: short Kirovohrad-Vinnytsia border
    "Kyiv City": ["Kyivska oblast"],
    "Kyivska oblast": ["Cherkaska oblast", "Chernihivska oblast", "Kyiv City", "Poltavska oblast",
                       "Vinnytska oblast", "Zhytomyrska oblast"],  # unsure: Kyiv oblast-Poltava (near Yahotyn)
    "Lvivska oblast": ["Ivano-Frankivska oblast", "Rivnenska oblast", "Ternopilska oblast", "Volynska oblast",
                       "Zakarpatska oblast"],
    "Mykolaivska oblast": ["Dnipropetrovska oblast", "Khersonska oblast", "Kirovohradska oblast", "Odeska oblast"],
    "Odeska oblast": ["Kirovohradska oblast", "Mykolaivska oblast", "Vinnytska oblast"],
    "Poltavska oblast": ["Cherkaska oblast", "Chernihivska oblast", "Dnipropetrovska oblast", "Kharkivska oblast",
                         "Kirovohradska oblast", "Kyivska oblast", "Sumska oblast"],
    "Rivnenska oblast": ["Khmelnytska oblast", "Lvivska oblast", "Ternopilska oblast", "Volynska oblast",
                         "Zhytomyrska oblast"],
    "Sumska oblast": ["Chernihivska oblast", "Kharkivska oblast", "Poltavska oblast"],
    "Ternopilska oblast": ["Chernivetska oblast", "Ivano-Frankivska oblast", "Khmelnytska oblast", "Lvivska oblast",
                           "Rivnenska oblast"],
    "Vinnytska oblast": ["Cherkaska oblast", "Chernivetska oblast", "Khmelnytska oblast", "Kirovohradska oblast",
                         "Kyivska oblast", "Odeska oblast", "Zhytomyrska oblast"],
    "Volynska oblast": ["Lvivska oblast", "Rivnenska oblast"],
    "Zakarpatska oblast": ["Ivano-Frankivska oblast", "Lvivska oblast"],
    "Zaporizka oblast": ["Dnipropetrovska oblast", "Donetska oblast", "Khersonska oblast"],
    "Zhytomyrska oblast": ["Khmelnytska oblast", "Kyivska oblast", "Rivnenska oblast", "Vinnytska oblast"],
}


def validate_adjacency(regions: list[str], adjacency: dict[str, list[str]] = ADJACENCY) -> None:
    """Assert names match the panel exactly, no self-loops, and the relation is symmetric."""
    assert set(adjacency) == set(regions), f"name mismatch: {set(adjacency) ^ set(regions)}"
    for r, neigh in adjacency.items():
        assert r not in neigh, f"self-loop: {r}"
        assert len(set(neigh)) == len(neigh), f"duplicate neighbour in {r}"
        for n in neigh:
            assert n in adjacency, f"unknown neighbour {n} of {r}"
            assert r in adjacency[n], f"asymmetric edge: {r} -> {n}"
