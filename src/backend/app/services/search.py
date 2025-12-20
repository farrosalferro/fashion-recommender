from ddgs import DDGS


def parse_search_results(search_results: dict[str, list[dict[str, str]]],) -> str:
    # Handle both dict and list inputs
    if isinstance(search_results, dict):
        items_to_parse = search_results.items()
    else:
        raise ValueError("search_results must be a dictionary")

    output_parts = []

    for item_name, results in items_to_parse:
        output = f"{item_name}:\n"

        for i, result in enumerate(results):
            result_num = i + 1
            # Calculate padding for alignment (accounts for "X. " prefix)
            num_width = len(str(result_num)) + 2  # +2 for ". "
            padding = " " * num_width

            for j, (key, value) in enumerate(result.items()):
                if j == 0:
                    output += f"\t{result_num}. {key}: {value}\n"
                else:
                    output += f"\t{padding}{key}: {value}\n"

        output_parts.append(output)

    return "\n".join(output_parts)


def search_item(items: list[str], max_results: int = 5) -> str:
    """ Search for items on internet

    Args:
        items: List of item names.
        max_results: Number of search results to return per item.

    Returns:
        A string of the title, link, and body of each search result for each item.
    """

    search_results = {}
    for item in items:
        search_results[item] = DDGS().text(item, max_results=max_results, timelimit="m", backend="google")
    return parse_search_results(search_results)
