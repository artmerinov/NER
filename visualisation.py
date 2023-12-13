from IPython.display import HTML

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

    return html_content.strip()  # Remove trailing space


# from IPython.display import HTML

# def get_unique_colors(entity_types):
#     # Predefined set of distinct colors
#     distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
#     # Assign a color to each entity type, repeat if necessary
#     color_mapping = {entity_type: distinct_colors[i % len(distinct_colors)] for i, entity_type in enumerate(entity_types)}
    
#     return color_mapping

# def highlight_entities(tokens, ner_tags, font_size=18):
#     highlighted_text = ""
#     current_entity = None
#     entity_colors = get_unique_colors(set(tag[2:] for tag in ner_tags if tag.startswith("B-")))

#     for token, tag in zip(tokens, ner_tags):
#         if tag.startswith("B-"):
#             if current_entity:
#                 highlighted_text += f"</span> ({current_entity}) "
#             entity_type = tag[2:]
#             color = entity_colors.get(entity_type, 'grey')  # Default to grey if the color is not found
#             highlighted_text += f"<span style='color:{color}'>{token} "
#             current_entity = entity_type
#         elif tag == "O":
#             if current_entity:
#                 highlighted_text += f"</span> ({current_entity}) {token} "
#                 current_entity = None
#             else:
#                 highlighted_text += f"{token} "
#         elif tag.startswith("I-"):
#             if current_entity:
#                 highlighted_text += f"{token} "
#             else:
#                 highlighted_text += f"{token} "

#     if current_entity:
#         highlighted_text += f"</span> ({current_entity}) "

#     # Highlight NER names in parenthesis
#     highlighted_text = highlighted_text.replace("(", "<span style='color:grey'>(").replace(")", ")</span>")

#     # Add overall style for font size
#     html_content = f"<div style='font-size:{font_size}px'>{highlighted_text}</div>"

#     return html_content.strip()  # Remove trailing space

# # Example usage
# tokens = ['The', 'Mona', 'Lisa', 'is', 'located', 'in', 'Paris', 'and', 'was', 'painted', 'by', 'Leonardo', 'da', 'Vinci', '.']
# ner_tags = ['O', 'B-art', 'I-art', 'O', 'O', 'O', 'B-location', 'O', 'O', 'O', 'O', 'B-person', 'I-person', 'I-person', 'O']

# highlighted_text = highlight_entities(tokens, ner_tags, font_size=20)  # Change the font size as needed
# HTML(highlighted_text)
