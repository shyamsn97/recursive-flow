// main.js
// Entry point for the app. Implements client-side behavior per shared requirements.
//
// Features implemented:
// - Wait for DOMContentLoaded before initializing.
// - Query and cache frequently-used DOM elements.
// - Provide a small module-style structure to avoid global pollution.
// - Implement a simple router to show/hide "pages" with data-route attributes.
// - Set up event delegation for clicks on elements with data-action attributes.
// - Provide utility functions: qs, qsa, on, ajax (fetch wrapper with JSON handling), debounce.
// - Expose an init() for manual re-initialization/testing and auto-init on load.
//
// Assumptions:
// - HTML has elements with data-route="<name>" for page sections.
// - Links/buttons that trigger actions have data-action="<name>" and optional data-target.
// - Forms that should be submitted via AJAX have data-ajax="true".
//
// Keep this file small and dependency-free.

(function () {
  'use strict';

  // Utilities
  var util = {
    qs: function (sel, ctx) { return (ctx || document).querySelector(sel); },
    qsa: function (sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); },
    on: function (el, evt, selOrHandler, handlerIfSel) {
      // on(el, 'click', handler) OR on(el, 'click', '.btn', handler)
      if (typeof selOrHandler === 'function') {
        el.addEventListener(evt, selOrHandler);
        return function () { el.removeEventListener(evt, selOrHandler); };
      }
      var sel = selOrHandler;
      var handler = handlerIfSel;
      var delegator = function (e) {
        var target = e.target;
        while (target && target !== el) {
          if (target.matches && target.matches(sel)) {
            handler.call(target, e);
            return;
          }
          target = target.parentNode;
        }
      };
      el.addEventListener(evt, delegator);
      return function () { el.removeEventListener(evt, delegator); };
    },
    ajax: function (url, opts) {
      opts = opts || {};
      var method = (opts.method || 'GET').toUpperCase();
      var headers = opts.headers || {};
      if (opts.json) {
        headers['Content-Type'] = headers['Content-Type'] || 'application/json';
      }
      var fetchOpts = {
        method: method,
        headers: headers,
        credentials: opts.credentials || 'same-origin'
      };
      if (opts.body != null) {
        fetchOpts.body = opts.json ? JSON.stringify(opts.body) : opts.body;
      }
      return fetch(url, fetchOpts).then(function (res) {
        var ctype = res.headers.get('content-type') || '';
        if (!res.ok) {
          // try to parse json error body
          if (ctype.indexOf('application/json') !== -1) {
            return res.json().then(function (j) { var err = new Error('Request failed'); err.status = res.status; err.body = j; throw err; });
          }
          var err = new Error('Request failed: ' + res.status);
          err.status = res.status;
          throw err;
        }
        if (ctype.indexOf('application/json') !== -1) {
          return res.json();
        }
        return res.text();
      });
    },
    debounce: function (fn, wait) {
      var t;
      return function () {
        var args = arguments;
        clearTimeout(t);
        t = setTimeout(function () { fn.apply(null, args); }, wait);
      };
    }
  };

  // Simple client-side router: show elements with data-route matching path, hide others
  var router = (function () {
    var routeAttr = 'data-route';
    function show(route) {
      var pages = util.qsa('[' + routeAttr + ']');
      pages.forEach(function (p) {
        if (p.getAttribute(routeAttr) === route) {
          p.style.display = '';
          p.setAttribute('aria-hidden', 'false');
        } else {
          p.style.display = 'none';
          p.setAttribute('aria-hidden', 'true');
        }
      });
      // update body attribute for styling/state
      document.body.setAttribute('data-current-route', route);
    }
    function init() {
      var hash = location.hash.replace(/^#/, '') || 'home';
      show(hash);
      // handle hashchange
      window.addEventListener('hashchange', function () {
        var r = location.hash.replace(/^#/, '') || 'home';
        show(r);
      });
    }
    return { init: init, show: show };
  }());

  // App initialization and event wiring
  var App = (function () {
    var delegates = [];

    function handleActionClick(ev) {
      var el = ev.currentTarget || ev.target;
      // walk up to element with data-action
      while (el && !el.getAttribute) el = el.parentNode;
      while (el && !el.getAttribute('data-action')) el = el.parentNode;
      if (!el) return;
      var action = el.getAttribute('data-action');
      var targetSelector = el.getAttribute('data-target');
      var payload = el.getAttribute('data-payload');
      // dispatch simple built-in actions
      if (action === 'navigate') {
        if (targetSelector) {
          location.hash = targetSelector.replace(/^#/, '');
        }
        return;
      }
      if (action === 'toggle') {
        var t = targetSelector ? util.qs(targetSelector) : el.nextElementSibling;
        if (t) {
          var vis = window.getComputedStyle(t).display !== 'none';
          t.style.display = vis ? 'none' : '';
        }
        return;
      }
      // custom events
      var customEvent = new CustomEvent('app:action', { detail: { action: action, target: targetSelector, payload: payload }, bubbles: true });
      el.dispatchEvent(customEvent);
    }

    function wireActionDelegation(root) {
      root = root || document;
      // delegate click for any element with data-action
      var remover = util.on(root, 'click', '[data-action]', function (e) {
        handleActionClick(e);
      });
      delegates.push(remover);
    }

    function wireAjaxForms(root) {
      root = root || document;
      var forms = util.qsa('form[data-ajax="true"]', root);
      forms.forEach(function (form) {
        if (form.__ajaxBound) return;
        form.addEventListener('submit', function (ev) {
          ev.preventDefault();
          var url = form.getAttribute('action') || location.href;
          var method = (form.getAttribute('method') || 'POST').toUpperCase();
          var formData = new FormData(form);
          // convert to JSON if data-json="true" or enctype is application/json
          var sendJson = form.getAttribute('data-json') === 'true' || form.enctype === 'application/json';
          var body, opts = { method: method };
          if (sendJson) {
            var obj = {};
            formData.forEach(function (v, k) {
              if (obj[k] !== undefined) {
                if (!Array.isArray(obj[k])) obj[k] = [obj[k]];
                obj[k].push(v);
              } else {
                obj[k] = v;
              }
            });
            opts.json = true;
            opts.body = obj;
          } else {
            opts.body = formData;
          }
          util.ajax(url, opts).then(function (res) {
            form.dispatchEvent(new CustomEvent('ajax:success', { detail: res }));
          }).catch(function (err) {
            form.dispatchEvent(new CustomEvent('ajax:error', { detail: err }));
          });
        });
        form.__ajaxBound = true;
      });
    }

    function init(root) {
      root = root || document;
      wireActionDelegation(root);
      wireAjaxForms(root);
      router.init();
    }

    function destroy() {
      delegates.forEach(function (rem) { try { rem(); } catch (e) {} });
      delegates = [];
      // remove any other listeners if we added them (none persistent besides hashchange)
    }

    return { init: init, destroy: destroy };
  }());

  // Auto-init on DOM ready
  function boot() {
    App.init(document);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    // already ready
    setTimeout(boot, 0);
  }

  // Expose to window for debugging/testing without polluting too many names
  window.App = window.App || App;
  window.appUtil = window.appUtil || util;

}());
