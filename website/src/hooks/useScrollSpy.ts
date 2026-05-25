import { useState, useEffect, useRef } from "react";

export function useScrollSpy(ids: string[], offset: number = 100) {
  const [activeId, setActiveId] = useState<string>("");

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + offset;
      
      let currentActiveId = "";
      for (const id of ids) {
        const element = document.getElementById(id);
        if (element) {
          const elementTop = element.getBoundingClientRect().top + window.scrollY;
          // Subtracted 20 pixels of bias to highlight right before hitting the exact border
          if (scrollPosition >= elementTop - 20) {
            currentActiveId = id;
          } else {
            break; // Since ids are ordered, we can break early
          }
        }
      }
      setActiveId(currentActiveId);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll(); // Initial check

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [ids, offset]);

  return activeId;
}
