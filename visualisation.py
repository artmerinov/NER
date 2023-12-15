from typing import List, Dict
import matplotlib.pyplot as plt


def plot_classification_report(tag_names: List[str], tag_scores: List[float]) -> None:

    plt.figure(figsize=(14, 4))
    bars = plt.bar(tag_names, tag_scores, color='slateblue', width=0.8, edgecolor='black')

    # annotate
    for bar, score in zip(bars, tag_scores):
        plt.text(
            x=bar.get_x() + bar.get_width() / 2, 
            y=bar.get_height() + 0.02, 
            s=f'{score:.2f}', 
            ha='center', 
            va='bottom',
            rotation=90,
            size=8
        )
    plt.xticks(rotation=90, size=8)
    plt.title('Precision of Tags')
    plt.xlabel('Tags')
    plt.ylabel('Precision')
    plt.ylim(0, 1)
    plt.show()


def get_tag_colors() -> Dict[str, str]:
    """
    Custom tag-color mapping.
    """
    tag_colors = {
        "art": '#1f77b4',
        "building": '#ff7f0e',
        "event": '#2ca02c',
        "location": '#d62728',
        "organization": '#9467bd',
        "other": '#8c564b',
        "person": '#e377c2',
        "product": '#7f7f7f'
    }
    return tag_colors


def highlight_entities(tokens: List[str], ner_tags: List[str], font_size: int = 18) -> str:
    """
    Outputs text with highlighted tags using HTML style.
    """
    highlighted_text = ""
    current_entity = None
    tag_colors = get_tag_colors()

    for token, tag in zip(tokens, ner_tags):
        if tag.startswith("B-"):
            if current_entity:
                highlighted_text += f"</span> ({current_entity}) "
            entity_type = tag[2:]
            color = tag_colors.get(entity_type, 'grey')
            highlighted_text += f"<span style='color:{color}'>{token} "
            current_entity = entity_type
        elif tag == "O":
            if current_entity:
                highlighted_text += f"</span> ({current_entity}) {token} "
                current_entity = None
            else:
                highlighted_text += f"{token} "
        elif tag.startswith("I-"):
            if current_entity:
                highlighted_text += f"{token} "
            else:
                highlighted_text += f"{token} "

    if current_entity:
        highlighted_text += f"</span> ({current_entity}) "

    # Highlight NER names in parenthesis
    highlighted_text = highlighted_text.replace("(", "<span style='color:grey'>(").replace(")", ")</span>")

    # Add overall style for font size
    html_content = f"<div style='font-size:{font_size}px'>{highlighted_text}</div>"

    return html_content.strip()