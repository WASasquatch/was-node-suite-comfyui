/**
 * The two scripts a Content Viewer frame runs while `viewer.allow_scripts` is off.
 *
 * Each function is written into the frame as its own source text, so it reads nothing
 * outside its own body.
 */

/**
 * Hold the frame to its own document and run only the scripts the viewer sends it.
 *
 * Cancels every navigation the frame starts that would leave its document, opens a link in a
 * new tab only once the person confirms the address, applies the parent's theme updates, and
 * runs the view scripts the parent posts under the frame's nonce.
 *
 * @param {{nodeId: string}} config - The node the frame belongs to.
 */
export function frameGuard(config) {
  var nonce = document.currentScript ? document.currentScript.nonce : "";
  var head = document.head;
  var xlink = "http://www.w3.org/1999/xlink";
  var themeStyle = null;
  window.WAS_NODE_ID = String(config && config.nodeId != null ? config.nodeId : "");
  window.WAS_SCRIPTS_LOADED = false;

  if (window.navigation && typeof window.navigation.addEventListener === "function") {
    window.navigation.addEventListener("navigate", function (event) {
      var destination = event.destination;
      if (destination && destination.sameDocument) return;
      // A download of a picture the view built itself, such as the canvas export.
      if (event.downloadRequest !== null && destination && /^(blob|data):/i.test(destination.url)) return;
      if (event.cancelable) event.preventDefault();
    });
  }

  function linkOf(target) {
    for (var node = target; node && node.nodeType === 1; node = node.parentNode) {
      var name = node.localName;
      if ((name === "a" || name === "area")
          && (node.hasAttribute("href") || node.hasAttributeNS(xlink, "href"))) {
        return node;
      }
    }
    return null;
  }

  function follow(event) {
    var link = linkOf(event.target);
    if (!link) return;
    var raw = String(link.getAttribute("href") || link.getAttributeNS(xlink, "href") || "").trim();
    if (link.hasAttribute("download") && /^(blob|data):/i.test(raw)) return;
    event.preventDefault();
    if (event.type !== "click") return;
    if (raw.charAt(0) === "#") {
      var id = raw.slice(1);
      try { id = decodeURIComponent(id); } catch (error) { return; }
      var anchor = id ? document.getElementById(id) : null;
      if (anchor && typeof anchor.scrollIntoView === "function") anchor.scrollIntoView();
      return;
    }
    var address;
    try { address = new URL(raw, document.baseURI); } catch (error) { return; }
    if (!/^(https?|mailto):$/.test(address.protocol)) return;
    if (window.confirm("Open this address in a new tab?\n\n" + address.href)) {
      window.open(address.href, "_blank", "noopener,noreferrer");
    }
  }

  window.addEventListener("click", follow, true);
  window.addEventListener("auxclick", follow, true);

  function run(code) {
    var script = document.createElement("script");
    script.nonce = nonce;
    script.textContent = code;
    head.appendChild(script);
    script.remove();
  }

  window.addEventListener("message", function (event) {
    if (event.source !== window.parent) return;
    var data = event.data;
    if (!data || typeof data !== "object") return;
    if (data.type === "was-theme-update") {
      if (!themeStyle) {
        themeStyle = document.createElement("style");
        head.appendChild(themeStyle);
      }
      themeStyle.textContent = String(data.cssVars || "");
      return;
    }
    if (data.type === "was-inject-scripts" && !window.WAS_SCRIPTS_LOADED) {
      window.WAS_SCRIPTS_LOADED = true;
      var scripts = Array.isArray(data.scripts) ? data.scripts : [];
      var inits = [];
      for (var i = 0; i < scripts.length; i++) {
        var one = scripts[i];
        if (!one || typeof one.code !== "string") continue;
        run(one.code);
        if (typeof one.init === "string") inits.push(one.init);
      }
      for (var j = 0; j < inits.length; j++) run(inits[j]);
    }
  });
}

/**
 * Put a node's content into the frame with nothing in it that runs or navigates.
 *
 * The content is parsed into a separate document that runs and fetches nothing, cut down to
 * the elements and attributes a page shows, searched a second time for anything that runs or
 * leaves the frame, and only then moved into this frame ahead of the script carrying it. A
 * note naming the setting is shown above it when anything was taken out.
 *
 * @param {string} encoded - The content as UTF-8, in base64.
 * @param {{fullDocument: boolean, note: string}} config - Whether the content is a whole
 *   page, whose head styles and root attributes are kept, and the note's text.
 */
export function placeContent(encoded, config) {
  var here = document.currentScript;
  var head = document.head;
  var body = document.body;
  var removed = 0;
  var set = function (names) {
    var table = Object.create(null);
    names.split(" ").forEach(function (name) { if (name) table[name] = true; });
    return table;
  };

  // Elements kept, by lower-case local name, across HTML, SVG and MathML.
  var ALLOWED = set(
    "a abbr acronym address area article aside audio b bdi bdo big blink blockquote body br "
    + "button canvas caption center cite code col colgroup data datalist dd del details dfn "
    + "dialog dir div dl dt em fieldset figcaption figure font footer h1 h2 h3 h4 h5 h6 head "
    + "header hgroup hr html i img input ins kbd label legend li main map mark marquee menu "
    + "meter nav nobr ol optgroup option output p picture pre progress q rp rt ruby s samp "
    + "search section select small source span strike strong style sub summary sup table "
    + "tbody td textarea tfoot th thead time title tr track tt u ul var video wbr "
    + "svg circle clippath defs desc ellipse filter g image line lineargradient marker mask "
    + "metadata path pattern polygon polyline radialgradient rect stop switch symbol text "
    + "textpath tspan use view foreignobject feblend fecolormatrix fecomponenttransfer "
    + "fecomposite feconvolvematrix fediffuselighting fedisplacementmap fedistantlight "
    + "fedropshadow feflood fefunca fefuncb fefuncg fefuncr fegaussianblur feimage femerge "
    + "femergenode femorphology feoffset fepointlight fespecularlighting fespotlight fetile "
    + "feturbulence "
    + "math menclose merror mfrac mi mmultiscripts mn mo mover mpadded mphantom mprescripts "
    + "mroot mrow ms mspace msqrt mstyle msub msubsup msup mtable mtd mtext mtr munder "
    + "munderover none semantics annotation"
  );
  // Elements taken out together with everything inside them.
  var DROPPED = set(
    "script noscript iframe frame frameset object embed applet base meta link template portal "
    + "fencedframe noembed noframes param set animate animatecolor animatemotion "
    + "animatetransform discard handler listener"
  );
  // Of those, the ones the note counts.
  var COUNTED = set("script noscript iframe frame frameset object embed applet base portal fencedframe");
  // Hyphenated names that are not custom elements, unwrapped like any unknown element.
  var RESERVED = set(
    "annotation-xml color-profile font-face font-face-src font-face-uri font-face-format "
    + "font-face-name missing-glyph"
  );
  // Attributes taken off every element.
  var STRIPPED = set("srcdoc formaction action form ping target http-equiv nonce is data codebase manifest lowsrc dynsrc");
  // Of those, the ones the note counts.
  var STRIPPED_COUNTED = set("srcdoc formaction action http-equiv");
  var LINKS = set("http https mailto");
  var MEDIA = set("http https blob data");
  var LINKING = set("a area");
  var PICTURING = set("image feimage");
  var ADDRESSED = set("href xlink:href src srcset poster background cite longdesc");

  function scheme(value) {
    var bare = String(value).replace(/[\x00-\x20\x7f-\x9f]+/g, "").toLowerCase();
    var found = /^([a-z][a-z0-9+.\-]*):/.exec(bare);
    return { bare: bare, name: found ? found[1] : "" };
  }

  function mediaAllowed(value) {
    var read = scheme(value);
    if (!read.name) return true;
    if (!MEDIA[read.name]) return false;
    return read.name !== "data" || /^data:(image|video|audio)\//.test(read.bare);
  }

  function addressAllowed(element, key, value) {
    var read = scheme(value);
    if (key === "href" || key === "xlink:href") {
      if (LINKING[element]) return !read.name || !!LINKS[read.name];
      if (PICTURING[element]) return mediaAllowed(value);
      return read.bare.charAt(0) === "#";
    }
    if (key === "srcset") {
      return String(value).split(",").every(function (candidate) {
        return mediaAllowed(candidate.trim().split(/\s+/)[0] || "");
      });
    }
    if (key === "cite" || key === "longdesc") return !read.name || !!LINKS[read.name];
    return mediaAllowed(value);
  }

  function scrub(element, name) {
    var attributes = Array.prototype.slice.call(element.attributes);
    for (var i = 0; i < attributes.length; i++) {
      var attribute = attributes[i];
      var key = attribute.name.toLowerCase();
      var value = attribute.value;
      var drop = false;
      if (key.slice(0, 2) === "on") {
        drop = true;
        removed++;
      } else if (STRIPPED[key]) {
        drop = true;
        if (STRIPPED_COUNTED[key]) removed++;
      } else if ((key === "name" || (key === "id" && name === "img")) && value in document) {
        drop = true;
      } else if (ADDRESSED[key] && !addressAllowed(name, key, value)) {
        drop = true;
        removed++;
      }
      if (drop) element.removeAttributeNode(attribute);
    }
  }

  function clean(parent) {
    var child = parent.firstChild;
    while (child) {
      var next = child.nextSibling;
      if (child.nodeType === 1) {
        var name = String(child.localName || "").toLowerCase();
        if (DROPPED[name]) {
          if (COUNTED[name] || (name === "meta" && child.hasAttribute("http-equiv"))) removed++;
          parent.removeChild(child);
        } else if (!ALLOWED[name] && (name.indexOf("-") < 0 || RESERVED[name])) {
          if (name === "form") removed++;
          clean(child);
          while (child.firstChild) parent.insertBefore(child.firstChild, child);
          parent.removeChild(child);
        } else {
          scrub(child, name);
          clean(child);
        }
      } else if (child.nodeType !== 3) {
        parent.removeChild(child);
      }
      child = next;
    }
  }

  // A second, separate search of what the first pass kept.
  function verify(root) {
    var found = root.querySelectorAll(
      "script, noscript, iframe, frame, frameset, object, embed, applet, base, meta, link, "
      + "template, form, portal, fencedframe, set, animate, annotation-xml"
    );
    for (var i = 0; i < found.length; i++) {
      if (found[i].parentNode) {
        found[i].parentNode.removeChild(found[i]);
        removed++;
      }
    }
    var every = [root].concat(Array.prototype.slice.call(root.querySelectorAll("*")));
    for (var j = 0; j < every.length; j++) {
      var attributes = Array.prototype.slice.call(every[j].attributes);
      for (var k = 0; k < attributes.length; k++) {
        var key = attributes[k].name.toLowerCase();
        var bare = scheme(attributes[k].value).bare;
        if (/^on/.test(key) || /^(srcdoc|formaction|action|http-equiv|target|ping)$/.test(key)
            || (/(^|:)(href|src)$/.test(key) && /^(javascript|vbscript|data:text)/.test(bare))) {
          every[j].removeAttributeNode(attributes[k]);
          removed++;
        }
      }
    }
  }

  function adopt(from, to, before) {
    while (from.firstChild) {
      var node = document.adoptNode(from.firstChild);
      to.insertBefore(node, before || null);
    }
  }

  function copyAttributes(from, to) {
    var attributes = Array.prototype.slice.call(from.attributes);
    for (var i = 0; i < attributes.length; i++) {
      try { to.setAttribute(attributes[i].name, attributes[i].value); } catch (error) { continue; }
    }
  }

  function showNote(text) {
    var host = document.createElement("div");
    host.setAttribute("style", "all: initial; display: block; margin: 0 0 8px 0;");
    var bar = document.createElement("div");
    bar.textContent = text;
    bar.setAttribute("role", "status");
    bar.setAttribute(
      "style",
      "display: block; padding: 6px 10px; border-radius: 4px; font: 12px/1.4 sans-serif; "
      + "color: var(--theme-fg, #ddd); background: rgba(127, 127, 127, 0.18); "
      + "border: 1px solid rgba(127, 127, 127, 0.4);"
    );
    if (typeof host.attachShadow === "function") {
      host.attachShadow({ mode: "closed" }).appendChild(bar);
    } else {
      host.appendChild(bar);
    }
    body.insertBefore(host, body.firstChild);
  }

  try {
    var binary = atob(String(encoded || ""));
    var bytes = new Uint8Array(binary.length);
    for (var b = 0; b < binary.length; b++) bytes[b] = binary.charCodeAt(b);
    var source = new TextDecoder().decode(bytes);
    var whole = !!(config && config.fullDocument);
    var parsed = new DOMParser().parseFromString(whole ? source : "<!DOCTYPE html><body>" + source, "text/html");
    var root = parsed.documentElement;
    scrub(root, "html");
    scrub(parsed.body, "body");
    clean(parsed.head);
    clean(parsed.body);
    verify(root);
    if (whole) {
      adopt(parsed.head, head, null);
      copyAttributes(root, document.documentElement);
      copyAttributes(parsed.body, body);
    }
    adopt(parsed.body, body, here);
  } catch (error) {
    var failed = document.createElement("pre");
    failed.textContent = "This content could not be shown: " + (error && error.message ? error.message : error);
    body.insertBefore(failed, here);
  }
  if (removed > 0 && config && config.note) showNote(String(config.note));
}
