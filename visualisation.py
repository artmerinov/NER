def get_tag_colors():
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


def highlight_entities(tokens, ner_tags, font_size=18):
    highlighted_text = ""
    current_entity = None
    tag_colors = get_tag_colors()

    for token, tag in zip(tokens, ner_tags):
        if tag.startswith("B-"):
            if current_entity:
                highlighted_text += f"</span> ({current_entity}) "
            entity_type = tag[2:]
            color = tag_colors.get(entity_type, 'grey')  # Default to grey if the color is not found
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