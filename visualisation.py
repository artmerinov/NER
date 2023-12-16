from typing import List, Dict
import matplotlib.pyplot as plt


def get_color_for_tag(tag: str) -> str:
    """
    Gets color based on tag name.
    """
    if 'art-' in tag:
        color = '#1f77b4'
    elif 'building-' in tag:
        color = '#ff7f0e'
    elif 'event-' in tag:
        color = '#2ca02c'
    elif 'location-' in tag:
        color = '#d62728'
    elif 'organization-' in tag:
        color = '#9467bd'
    elif 'other-' in tag:
        color = '#8c564b'
    elif 'person-' in tag:
        color = '#e377c2'
    elif 'product-' in tag:
        color = '#7f7f7f'
    else:
        color = 'dimgray'

    return color


def plot_classification_report(tag_names: List[str], tag_scores: List[float]) -> None:
    """
    Plots bar chart for tag names and its precitions.
    """
    plt.figure(figsize=(14, 4))
    bar_colors = [get_color_for_tag(tag) for tag in tag_names]
    bars = plt.bar(tag_names, tag_scores, color=bar_colors, width=0.8, edgecolor='black', alpha=0.9)

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


def highlight_entities(tokens: List[str], ner_tags: List[str], font_size: int = 18) -> str:
    """
    Outputs text in html style in order to highlight tags.
    """
    text = ""
    prev_entity = None

    for token, tag in zip(tokens, ner_tags):
        if tag.startswith("B-"):
            if prev_entity:
                # previous word is finished TAG (because curr TAG is "B")
                # --> insert TAG name in parenthesis before new word 
                text += f"</span> ({prev_entity}) {token} "
            curr_entity = tag[2:]
            color = get_color_for_tag(curr_entity)
            text += f"<span style='color:{color}'>{token} "
            prev_entity = curr_entity
        elif tag == "O":
            if prev_entity:
                # previous word is finished TAG (because curr TAG is "O")
                # --> insert TAG name in parenthesis before new word
                text += f"</span> ({prev_entity}) {token} "
                prev_entity = None
            else:
                text += f"{token} "
        elif tag.startswith("I-"):
            text += f"{token} "

    if prev_entity:
        text += f"</span> ({prev_entity}) "

    # Highlight NER names in parenthesis
    text = text.replace("(", "<span style='color:grey'>(").replace(")", ")</span>")

    # Add overall style for font size
    html_content = f"<div style='font-size:{font_size}px'>{text}</div>"

    return html_content.strip()