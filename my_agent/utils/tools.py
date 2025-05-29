from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from typing import Dict, List, Optional

@tool
def draft_email(recipient: str, subject: str, key_points: str) -> str:
    """Draft a professional email with the given recipient, subject, and key points.
    
    Args:
        recipient: The email recipient's name or email address
        subject: The subject line of the email
        key_points: The main points to include in the email body
        
    Returns:
        A professionally formatted email draft
    """
    email_template = f"""Dear {recipient},

{key_points}

Best regards,
[Your name]"""
    
    return email_template

@tool
def compare_material_specs(
    requested_specs: Dict[str, any],
    received_specs: Dict[str, any],
    tolerance_rules: Optional[Dict[str, float]] = None
) -> Dict[str, any]:
    """Compare building material specifications between requested and received items.
    
    Args:
        requested_specs: Dictionary containing the original specifications requested
            Expected keys:
            - dimensions: Dict with length, width, height, etc.
            - quantity: Number of items
            - technical_specs: Dict of technical specifications
            - standards: List of applicable standards
            - tolerances: Dict of acceptable tolerances
            - compliance: List of compliance requirements
        received_specs: Dictionary containing the specifications received from supplier
            Same structure as requested_specs
        tolerance_rules: Optional dictionary of custom tolerance rules for specific parameters
            
    Returns:
        Dictionary containing:
        - matches: List of specifications that match
        - mismatches: List of specifications that don't match
        - warnings: List of specifications that are close to tolerance limits
        - compliance_status: Overall compliance status
    """
    results = {
        "matches": [],
        "mismatches": [],
        "warnings": [],
        "compliance_status": "PENDING"
    }
    
    # Compare dimensions
    if "dimensions" in requested_specs and "dimensions" in received_specs:
        for dim, req_value in requested_specs["dimensions"].items():
            if dim in received_specs["dimensions"]:
                rec_value = received_specs["dimensions"][dim]
                tolerance = tolerance_rules.get(dim, 0.01) if tolerance_rules else 0.01
                
                if abs(req_value - rec_value) <= tolerance:
                    results["matches"].append(f"Dimension {dim}: {rec_value} (within tolerance)")
                else:
                    results["mismatches"].append(f"Dimension {dim}: requested {req_value}, received {rec_value}")
    
    # Compare quantity
    if "quantity" in requested_specs and "quantity" in received_specs:
        if requested_specs["quantity"] == received_specs["quantity"]:
            results["matches"].append(f"Quantity: {received_specs['quantity']}")
        else:
            results["mismatches"].append(f"Quantity: requested {requested_specs['quantity']}, received {received_specs['quantity']}")
    
    # Compare technical specifications
    if "technical_specs" in requested_specs and "technical_specs" in received_specs:
        for spec, req_value in requested_specs["technical_specs"].items():
            if spec in received_specs["technical_specs"]:
                rec_value = received_specs["technical_specs"][spec]
                if req_value == rec_value:
                    results["matches"].append(f"Technical spec {spec}: {rec_value}")
                else:
                    results["mismatches"].append(f"Technical spec {spec}: requested {req_value}, received {rec_value}")
    
    # Compare standards
    if "standards" in requested_specs and "standards" in received_specs:
        req_standards = set(requested_specs["standards"])
        rec_standards = set(received_specs["standards"])
        
        if req_standards.issubset(rec_standards):
            results["matches"].append("All required standards are met")
        else:
            missing = req_standards - rec_standards
            results["mismatches"].append(f"Missing standards: {', '.join(missing)}")
    
    # Compare compliance requirements
    if "compliance" in requested_specs and "compliance" in received_specs:
        req_compliance = set(requested_specs["compliance"])
        rec_compliance = set(received_specs["compliance"])
        
        if req_compliance.issubset(rec_compliance):
            results["matches"].append("All compliance requirements are met")
        else:
            missing = req_compliance - rec_compliance
            results["mismatches"].append(f"Missing compliance requirements: {', '.join(missing)}")
    
    # Determine overall compliance status
    if not results["mismatches"]:
        results["compliance_status"] = "COMPLIANT"
    elif len(results["mismatches"]) > len(results["matches"]):
        results["compliance_status"] = "NON-COMPLIANT"
    else:
        results["compliance_status"] = "PARTIALLY COMPLIANT"
    
    return results

tools = [TavilySearchResults(max_results=4), draft_email, compare_material_specs]